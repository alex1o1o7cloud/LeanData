import Mathlib

namespace dress_price_calculation_l373_37302

/-- Given a dress with an original price, discount rate, and tax rate, 
    calculate the total selling price after discount and tax. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let taxAmount := salePrice * taxRate
  salePrice + taxAmount

/-- Theorem stating that for a dress with original price $80, 25% discount, 
    and 10% tax, the total selling price is $66. -/
theorem dress_price_calculation :
  totalSellingPrice 80 0.25 0.10 = 66 := by
  sorry

#eval totalSellingPrice 80 0.25 0.10

end dress_price_calculation_l373_37302


namespace smallest_divisible_by_10_13_14_l373_37355

theorem smallest_divisible_by_10_13_14 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 10 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end smallest_divisible_by_10_13_14_l373_37355


namespace range_of_f_l373_37331

-- Define the function f
def f (x : ℝ) : ℝ := |x + 10| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-15) 25 := by sorry

end range_of_f_l373_37331


namespace parabola_tangent_angle_sine_l373_37301

/-- Given a parabola x^2 = 4y with focus F(0, 1), and a point A on the parabola where the tangent line has slope 2, 
    prove that the sine of the angle between AF and the tangent line at A is √5/5. -/
theorem parabola_tangent_angle_sine (A : ℝ × ℝ) : 
  let (x, y) := A
  (x^2 = 4*y) →                   -- A is on the parabola
  ((1/2)*x = 2) →                 -- Slope of tangent at A is 2
  let F := (0, 1)                 -- Focus of the parabola
  let slope_AF := (y - 1) / (x - 0)
  let tan_theta := |((1/2)*x - slope_AF) / (1 + (1/2)*x * slope_AF)|
  Real.sqrt (tan_theta^2 / (1 + tan_theta^2)) = Real.sqrt 5 / 5 :=
by sorry

end parabola_tangent_angle_sine_l373_37301


namespace dove_population_growth_l373_37389

theorem dove_population_growth (initial_doves : ℕ) (eggs_per_dove : ℕ) (hatch_rate : ℚ) : 
  initial_doves = 20 →
  eggs_per_dove = 3 →
  hatch_rate = 3/4 →
  initial_doves + (initial_doves * eggs_per_dove * hatch_rate).floor = 65 :=
by sorry

end dove_population_growth_l373_37389


namespace certain_number_proof_l373_37323

theorem certain_number_proof : ∃ n : ℕ, n - 999 = 9001 ∧ n = 10000 := by
  sorry

end certain_number_proof_l373_37323


namespace max_profit_theorem_l373_37358

/-- Represents the profit function for a product given its price increase -/
def profit_function (x : ℕ) : ℝ := -10 * x^2 + 170 * x + 2100

/-- Represents the constraint on the price increase -/
def price_increase_constraint (x : ℕ) : Prop := 0 < x ∧ x ≤ 15

theorem max_profit_theorem :
  ∃ (x : ℕ), price_increase_constraint x ∧
    (∀ (y : ℕ), price_increase_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 2400 ∧
    (x = 5 ∨ x = 6) := by sorry

end max_profit_theorem_l373_37358


namespace mean_temperature_l373_37354

def temperatures : List ℚ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 770 / 9 := by sorry

end mean_temperature_l373_37354


namespace six_boxes_consecutive_green_balls_l373_37310

/-- The number of ways to fill n boxes with red or green balls, such that at least one box
    contains a green ball and the boxes containing green balls are consecutively numbered. -/
def consecutiveGreenBalls (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that for 6 boxes, there are 21 ways to fill them under the given conditions. -/
theorem six_boxes_consecutive_green_balls :
  consecutiveGreenBalls 6 = 21 := by
  sorry

end six_boxes_consecutive_green_balls_l373_37310


namespace chocolate_triangles_l373_37371

theorem chocolate_triangles (square_side : ℝ) (triangle_width : ℝ) (triangle_height : ℝ)
  (h_square : square_side = 10)
  (h_width : triangle_width = 1)
  (h_height : triangle_height = 3) :
  ⌊(square_side^2) / ((triangle_width * triangle_height) / 2)⌋ = 66 := by
  sorry

end chocolate_triangles_l373_37371


namespace spider_dressing_combinations_l373_37337

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of socks per leg -/
def socks_per_leg : ℕ := 2

/-- The number of shoes per leg -/
def shoes_per_leg : ℕ := 1

/-- The total number of items to wear -/
def total_items : ℕ := num_legs * (socks_per_leg + shoes_per_leg)

/-- The number of ways to arrange socks on one leg -/
def sock_arrangements_per_leg : ℕ := 2  -- 2! = 2

theorem spider_dressing_combinations :
  (Nat.choose total_items num_legs) * (sock_arrangements_per_leg ^ num_legs) =
  (Nat.factorial total_items) / (Nat.factorial num_legs * Nat.factorial (total_items - num_legs)) * 1024 :=
by sorry

end spider_dressing_combinations_l373_37337


namespace square_root_sum_equals_ten_l373_37365

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end square_root_sum_equals_ten_l373_37365


namespace common_factor_proof_l373_37392

def expression (x y : ℝ) : ℝ := 9 * x^3 * y^2 + 12 * x^2 * y^3

def common_factor (x y : ℝ) : ℝ := 3 * x^2 * y^2

theorem common_factor_proof :
  ∀ x y : ℝ, ∃ k : ℝ, expression x y = common_factor x y * k :=
by sorry

end common_factor_proof_l373_37392


namespace cube_volume_from_surface_area_l373_37313

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → (6 * s^2 = 54) → s^3 = 27 := by
  sorry

end cube_volume_from_surface_area_l373_37313


namespace biloca_path_theorem_l373_37330

/-- Represents the dimensions and paths of ants on a tiled floor -/
structure AntPaths where
  diagonal_length : ℝ
  tile_width : ℝ
  tile_length : ℝ
  pipoca_path : ℝ
  tonica_path : ℝ
  cotinha_path : ℝ

/-- Calculates the length of Biloca's path -/
def biloca_path_length (ap : AntPaths) : ℝ :=
  3 * ap.diagonal_length + 4 * ap.tile_width + 2 * ap.tile_length

/-- Theorem stating the length of Biloca's path -/
theorem biloca_path_theorem (ap : AntPaths) 
  (h1 : ap.pipoca_path = 5 * ap.diagonal_length)
  (h2 : ap.pipoca_path = 25)
  (h3 : ap.tonica_path = 5 * ap.diagonal_length + 4 * ap.tile_width)
  (h4 : ap.tonica_path = 37)
  (h5 : ap.cotinha_path = 5 * ap.tile_length + 4 * ap.tile_width)
  (h6 : ap.cotinha_path = 32) :
  biloca_path_length ap = 35 := by
  sorry


end biloca_path_theorem_l373_37330


namespace roots_sum_and_product_inequality_l373_37399

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem roots_sum_and_product_inequality 
  (x₁ x₂ : ℝ) 
  (h_pos₁ : x₁ > 0) 
  (h_pos₂ : x₂ > 0) 
  (h_distinct : x₁ ≠ x₂) 
  (h_root₁ : f x₁ = 3 * Real.exp 1 * x₁ + 3 * Real.exp 1 * Real.log x₁)
  (h_root₂ : f x₂ = 3 * Real.exp 1 * x₂ + 3 * Real.exp 1 * Real.log x₂) :
  x₁ + x₂ + Real.log (x₁ * x₂) > 2 := by
  sorry

end roots_sum_and_product_inequality_l373_37399


namespace parallelepiped_dimensions_l373_37352

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2 : ℕ) > 0 ∧ (n - 4 : ℕ) > 0 ∧
  (n - 2 : ℕ) * (n - 4 : ℕ) * (n - 6 : ℕ) = 2 * n * (n - 2 : ℕ) * (n - 4 : ℕ) / 3 →
  n = 18 := by sorry

end parallelepiped_dimensions_l373_37352


namespace certain_number_proof_l373_37303

theorem certain_number_proof (x : ℝ) : x * 2.13 = 0.3408 → x = 0.1600 := by
  sorry

end certain_number_proof_l373_37303


namespace count_valid_primes_l373_37353

def isSubnumber (n m : ℕ) : Prop :=
  ∃ (k l : ℕ), n = (m / 10^k) % (10^l)

def hasNonPrimeSubnumber (n : ℕ) : Prop :=
  ∃ (m : ℕ), isSubnumber m n ∧ m > 1 ∧ ¬ Nat.Prime m

def validPrime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n < 1000000000 ∧ ¬ hasNonPrimeSubnumber n

theorem count_valid_primes :
  ∃! (s : Finset ℕ), (∀ n ∈ s, validPrime n) ∧ s.card = 9 :=
sorry

end count_valid_primes_l373_37353


namespace geometric_sequence_proof_l373_37369

def geometric_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem geometric_sequence_proof (b : ℝ) 
  (h₁ : geometric_sequence 150 b (60/36)) 
  (h₂ : b > 0) : 
  b = 5 * Real.sqrt 10 ∧ ¬ arithmetic_sequence 150 b (60/36) := by
  sorry

#check geometric_sequence_proof

end geometric_sequence_proof_l373_37369


namespace quadratic_transformation_l373_37390

theorem quadratic_transformation (x : ℝ) : 
  (2 * x^2 - 3 * x + 1 = 0) ↔ ((x - 3/4)^2 = 1/16) :=
by sorry

end quadratic_transformation_l373_37390


namespace problem_solution_l373_37339

theorem problem_solution (a b : ℝ) (h : |a - 1| + Real.sqrt (b + 2) = 0) : 
  (a + b) ^ 2022 = 1 := by
  sorry

end problem_solution_l373_37339


namespace no_integer_solutions_to_equation_l373_37320

theorem no_integer_solutions_to_equation :
  ¬∃ (w x y z : ℤ), (5 : ℝ)^w + (5 : ℝ)^x = (7 : ℝ)^y + (7 : ℝ)^z :=
by sorry

end no_integer_solutions_to_equation_l373_37320


namespace trig_expression_equals_neg_sqrt_three_l373_37345

theorem trig_expression_equals_neg_sqrt_three : 
  (2 * Real.sin (10 * π / 180) - Real.cos (20 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end trig_expression_equals_neg_sqrt_three_l373_37345


namespace line_l_equation_l373_37346

/-- The fixed point A through which the line mx - y - m + 2 = 0 always passes -/
def A : ℝ × ℝ := (1, 2)

/-- The slope of the line 2x + y - 2 = 0 -/
def k : ℝ := -2

/-- The equation of the line l passing through A and parallel to 2x + y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

theorem line_l_equation : ∀ m : ℝ, 
  (m * A.1 - A.2 - m + 2 = 0) → 
  (∀ x y : ℝ, line_l x y ↔ y - A.2 = k * (x - A.1)) :=
by sorry

end line_l_equation_l373_37346


namespace triangle_problem_l373_37359

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c →
  b + c = 5 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end triangle_problem_l373_37359


namespace square_perimeter_from_rectangle_perimeter_l373_37395

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 32 inches, then the perimeter of the square is 51.2 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) 
  (h1 : s > 0) 
  (h2 : 2 * s + 2 * (s / 4) = 32) : 
  4 * s = 51.2 := by
  sorry

end square_perimeter_from_rectangle_perimeter_l373_37395


namespace solution_set_quadratic_inequality_l373_37376

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 2 * x + 8
  {x : ℝ | f x > 0} = Set.Ioo (-4/3 : ℝ) 2 := by sorry

end solution_set_quadratic_inequality_l373_37376


namespace power_mod_seven_l373_37367

theorem power_mod_seven : 3^87 + 5 ≡ 4 [ZMOD 7] := by sorry

end power_mod_seven_l373_37367


namespace arrangement_count_l373_37321

def number_of_arrangements (black red blue : ℕ) : ℕ :=
  Nat.factorial (black + red + blue) / (Nat.factorial black * Nat.factorial red * Nat.factorial blue)

theorem arrangement_count :
  number_of_arrangements 2 3 4 = 1260 := by
  sorry

end arrangement_count_l373_37321


namespace dogsled_race_distance_l373_37393

/-- The distance of the dogsled race course -/
def distance : ℝ := sorry

/-- The time taken by Team W to complete the course -/
def time_W : ℝ := sorry

/-- The time taken by Team A to complete the course -/
def time_A : ℝ := sorry

/-- The average speed of Team W -/
def speed_W : ℝ := 20

/-- The average speed of Team A -/
def speed_A : ℝ := speed_W + 5

theorem dogsled_race_distance :
  (time_A = time_W - 3) →
  (distance = speed_W * time_W) →
  (distance = speed_A * time_A) →
  distance = 300 := by sorry

end dogsled_race_distance_l373_37393


namespace savings_percentage_l373_37326

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- The financial situation satisfies the given conditions --/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = 1.5 * fs.income_year1 ∧
  fs.savings_year2 = 2 * fs.savings_year1 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- The theorem stating that the man saved 50% of his income in the first year --/
theorem savings_percentage (fs : FinancialSituation) (h : satisfies_conditions fs) :
  fs.savings_year1 / fs.income_year1 = 0.5 := by
  sorry

end savings_percentage_l373_37326


namespace common_root_inequality_l373_37306

theorem common_root_inequality (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 1)
  (eq1 : t^2 + a*t - 100 = 0) (eq2 : t^2 - 200*t + b = 0) : b - a > 100 := by
  sorry

end common_root_inequality_l373_37306


namespace average_temperature_l373_37357

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature : 
  (temperatures.sum : ℚ) / temperatures.length = -12 := by
  sorry

end average_temperature_l373_37357


namespace total_inches_paved_before_today_l373_37328

/-- Represents a road section with its length and completion percentage -/
structure RoadSection where
  length : ℝ
  percentComplete : ℝ

/-- Calculates the total inches repaved before today given three road sections and additional inches repaved today -/
def totalInchesPavedBeforeToday (sectionA sectionB sectionC : RoadSection) (additionalInches : ℝ) : ℝ :=
  sectionA.length * sectionA.percentComplete +
  sectionB.length * sectionB.percentComplete +
  sectionC.length * sectionC.percentComplete

/-- Theorem stating that the total inches repaved before today is 6900 -/
theorem total_inches_paved_before_today :
  let sectionA : RoadSection := { length := 4000, percentComplete := 0.7 }
  let sectionB : RoadSection := { length := 3500, percentComplete := 0.6 }
  let sectionC : RoadSection := { length := 2500, percentComplete := 0.8 }
  let additionalInches : ℝ := 950
  totalInchesPavedBeforeToday sectionA sectionB sectionC additionalInches = 6900 := by
  sorry

end total_inches_paved_before_today_l373_37328


namespace problem_statement_l373_37377

theorem problem_statement (n : ℕ+) 
  (h1 : ∃ a : ℕ+, (3 * n + 1 : ℕ) = a ^ 2)
  (h2 : ∃ b : ℕ+, (5 * n - 1 : ℕ) = b ^ 2) :
  (∃ p q : ℕ+, p * q = 7 * n + 13 ∧ p ≠ 1 ∧ q ≠ 1) ∧
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) :=
by sorry

end problem_statement_l373_37377


namespace complex_modulus_problem_l373_37311

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l373_37311


namespace intersection_angle_l373_37336

-- Define the lines
def line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def line2 (x : ℝ) : Prop := x + 5 = 0

-- Define the angle between the lines
def angle_between_lines : ℝ := 30

-- Theorem statement
theorem intersection_angle :
  ∃ (x y : ℝ), line1 x y ∧ line2 x → angle_between_lines = 30 := by sorry

end intersection_angle_l373_37336


namespace no_fixed_points_l373_37322

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

/-- The specific function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ :=
  x^2 + 1

/-- Theorem: f(x) = x^2 + 1 has no fixed points -/
theorem no_fixed_points : ¬∃ x : ℝ, is_fixed_point f x := by
  sorry

end no_fixed_points_l373_37322


namespace grazing_area_difference_l373_37362

/-- Proves that the area difference between two circular grazing arrangements is 35π square feet -/
theorem grazing_area_difference (rope_length : ℝ) (tank_radius : ℝ) : 
  rope_length = 12 → tank_radius = 10 → 
  π * rope_length^2 - (3/4 * π * rope_length^2 + 1/4 * π * (rope_length - tank_radius)^2) = 35 * π := by
  sorry

end grazing_area_difference_l373_37362


namespace sqrt_square_equals_abs_l373_37350

theorem sqrt_square_equals_abs (a : ℝ) : Real.sqrt (a^2) = |a| := by
  sorry

end sqrt_square_equals_abs_l373_37350


namespace tulip_fraction_l373_37347

theorem tulip_fraction (total : ℕ) (yellow_ratio red_ratio pink_ratio : ℚ) : 
  total = 60 ∧
  yellow_ratio = 1/2 ∧
  red_ratio = 1/3 ∧
  pink_ratio = 1/4 →
  (total - (yellow_ratio * total) - 
   (red_ratio * (total - yellow_ratio * total)) - 
   (pink_ratio * (total - yellow_ratio * total - red_ratio * (total - yellow_ratio * total)))) / total = 1/4 :=
by sorry

end tulip_fraction_l373_37347


namespace letters_in_mailboxes_l373_37338

/-- The number of ways to distribute n items into k categories -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of letters -/
def num_letters : ℕ := 4

/-- The number of mailboxes -/
def num_mailboxes : ℕ := 3

/-- Theorem: The number of ways to put 4 letters into 3 mailboxes is 81 -/
theorem letters_in_mailboxes :
  distribute num_letters num_mailboxes = 81 := by sorry

end letters_in_mailboxes_l373_37338


namespace julians_comic_frames_l373_37319

/-- The number of frames on each page of Julian's comic book -/
def frames_per_page : ℕ := 11

/-- The number of pages in Julian's comic book -/
def total_pages : ℕ := 13

/-- The total number of frames in Julian's comic book -/
def total_frames : ℕ := frames_per_page * total_pages

theorem julians_comic_frames :
  total_frames = 143 := by
  sorry

end julians_comic_frames_l373_37319


namespace committee_count_is_738_l373_37388

/-- Represents a department in the university's science division -/
inductive Department
| Physics
| Chemistry
| Biology

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of professors in each department for each gender -/
def professors_per_dept_gender : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 7

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 3

/-- The number of professors required from the physics department -/
def required_physics : Nat := 3

/-- The number of professors required from each of chemistry and biology departments -/
def required_chem_bio : Nat := 2

/-- Calculates the number of possible committees given the conditions -/
def count_committees (professors : List Professor) : Nat :=
  sorry

/-- Theorem stating that the number of possible committees is 738 -/
theorem committee_count_is_738 (professors : List Professor) : 
  count_committees professors = 738 := by
  sorry

end committee_count_is_738_l373_37388


namespace sandwich_count_l373_37300

def num_meats : ℕ := 12
def num_cheeses : ℕ := 8
def num_toppings : ℕ := 5

def sandwich_combinations : ℕ := num_meats * (num_cheeses.choose 2) * num_toppings

theorem sandwich_count : sandwich_combinations = 1680 := by
  sorry

end sandwich_count_l373_37300


namespace described_loop_is_while_loop_l373_37312

/-- Represents a generic loop structure -/
structure LoopStructure :=
  (condition_evaluation : Bool)
  (execution_order : Bool)

/-- Defines a While loop structure -/
def is_while_loop (loop : LoopStructure) : Prop :=
  loop.condition_evaluation ∧ loop.execution_order

/-- Theorem stating that the described loop structure is a While loop -/
theorem described_loop_is_while_loop :
  ∀ (loop : LoopStructure),
  loop.condition_evaluation = true ∧
  loop.execution_order = true →
  is_while_loop loop :=
by
  sorry

#check described_loop_is_while_loop

end described_loop_is_while_loop_l373_37312


namespace sixth_grade_students_l373_37378

/-- The number of students in the sixth grade -/
def total_students : ℕ := 147

/-- The number of books available -/
def total_books : ℕ := 105

/-- The number of boys in the sixth grade -/
def num_boys : ℕ := 84

/-- The number of girls in the sixth grade -/
def num_girls : ℕ := 63

theorem sixth_grade_students :
  (total_students = num_boys + num_girls) ∧
  (total_books = 105) ∧
  (num_boys + (num_girls / 3) = total_books) ∧
  (num_girls + (num_boys / 2) = total_books) :=
by sorry

end sixth_grade_students_l373_37378


namespace geometric_sequence_sum_l373_37341

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_4 = 24, prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 3) (h_a4 : a 4 = 24) : a 3 + a 4 + a 5 = 84 := by
  sorry

#check geometric_sequence_sum

end geometric_sequence_sum_l373_37341


namespace expenditure_ratio_l373_37374

theorem expenditure_ratio (anand_income balu_income anand_expenditure balu_expenditure : ℚ) :
  anand_income / balu_income = 5 / 4 →
  anand_income = 2000 →
  anand_income - anand_expenditure = 800 →
  balu_income - balu_expenditure = 800 →
  anand_expenditure / balu_expenditure = 3 / 2 := by
sorry

end expenditure_ratio_l373_37374


namespace train_speed_problem_l373_37317

/-- Proves that given the conditions of the train problem, the speeds of the slower and faster trains are 60 km/hr and 70 km/hr respectively. -/
theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (remaining_distance : ℝ)
  (h1 : distance = 300)
  (h2 : time = 2)
  (h3 : speed_diff = 10)
  (h4 : remaining_distance = 40) :
  ∃ (v1 v2 : ℝ), v1 = 60 ∧ v2 = 70 ∧ v2 = v1 + speed_diff ∧
  distance - remaining_distance = (v1 + v2) * time :=
by sorry

end train_speed_problem_l373_37317


namespace rotten_apples_l373_37334

/-- Given a problem about apples in crates and boxes, prove the number of rotten apples. -/
theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (apples_per_box : ℕ) (num_boxes : ℕ)
  (h1 : apples_per_crate = 42)
  (h2 : num_crates = 12)
  (h3 : apples_per_box = 10)
  (h4 : num_boxes = 50) :
  apples_per_crate * num_crates - apples_per_box * num_boxes = 4 := by
  sorry

#check rotten_apples

end rotten_apples_l373_37334


namespace prime_sum_24_l373_37372

theorem prime_sum_24 (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c → a * b + b * c = 119 → a + b + c = 24 := by
sorry

end prime_sum_24_l373_37372


namespace sandwich_combinations_l373_37397

def num_toppings : ℕ := 10
def num_patty_types : ℕ := 3

theorem sandwich_combinations :
  (2^num_toppings) * num_patty_types = 3072 := by sorry

end sandwich_combinations_l373_37397


namespace mistaken_quotient_l373_37356

theorem mistaken_quotient (D : ℕ) : 
  D % 21 = 0 ∧ D / 21 = 20 → D / 12 = 35 := by
  sorry

end mistaken_quotient_l373_37356


namespace max_area_rectangular_enclosure_l373_37380

/-- The maximum area of a rectangular enclosure with given constraints -/
theorem max_area_rectangular_enclosure 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 400) 
  (h_min_length : min_length = 100) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      length * width ≥ l * w ∧ 
      length * width = 10000 :=
sorry

end max_area_rectangular_enclosure_l373_37380


namespace angle_conversion_l373_37385

theorem angle_conversion :
  ∃ (k : ℤ) (α : ℝ), -1485 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  use -5
  use 315
  sorry

end angle_conversion_l373_37385


namespace shopping_money_l373_37351

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 2800 → initial_amount = 4000 := by
  sorry

end shopping_money_l373_37351


namespace midpoint_trajectory_l373_37348

/-- The trajectory of the midpoint of a line segment between a point on a circle and a fixed point -/
theorem midpoint_trajectory (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 4 →  -- P is on the circle x^2 + y^2 = 4
  x = (x₀ + 8) / 2 →  -- x-coordinate of midpoint M
  y = y₀ / 2 →  -- y-coordinate of midpoint M
  (x - 4)^2 + y^2 = 1 :=  -- Trajectory equation
by sorry

end midpoint_trajectory_l373_37348


namespace total_payment_example_l373_37309

/-- Calculates the total amount paid for a meal including sales tax and tip -/
def total_payment (meal_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  meal_cost * (1 + sales_tax_rate + tip_rate)

/-- Theorem: The total payment for a $100 meal with 4% sales tax and 6% tip is $110 -/
theorem total_payment_example : total_payment 100 0.04 0.06 = 110 := by
  sorry

end total_payment_example_l373_37309


namespace max_value_of_g_l373_37327

def g (x : ℝ) := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end max_value_of_g_l373_37327


namespace lucky_in_thirteen_l373_37324

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop :=
  sum_of_digits n % 7 = 0

/-- Main theorem: Any sequence of 13 consecutive natural numbers contains a lucky number -/
theorem lucky_in_thirteen (start : ℕ) : ∃ k : ℕ, k ∈ Finset.range 13 ∧ is_lucky (start + k) := by
  sorry

end lucky_in_thirteen_l373_37324


namespace target_number_scientific_notation_l373_37344

/-- The number we want to express in scientific notation -/
def target_number : ℕ := 1200000000

/-- Definition of scientific notation for positive integers -/
def scientific_notation (n : ℕ) (a : ℚ) (b : ℤ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (n = (a * 10^b).floor)

/-- Theorem stating that 1,200,000,000 is equal to 1.2 × 10^9 in scientific notation -/
theorem target_number_scientific_notation :
  scientific_notation target_number (12/10) 9 := by
  sorry

end target_number_scientific_notation_l373_37344


namespace election_win_percentage_l373_37364

/-- In a two-candidate election, if a candidate receives 45% of the total votes,
    they need more than 50% of the total votes to win. -/
theorem election_win_percentage (total_votes : ℕ) (candidate_votes : ℕ) 
    (h1 : candidate_votes = (45 : ℕ) * total_votes / 100) 
    (h2 : total_votes > 0) : 
    ∃ (winning_percentage : ℚ), 
      winning_percentage > (1 : ℚ) / 2 ∧ 
      winning_percentage * total_votes > candidate_votes := by
  sorry

end election_win_percentage_l373_37364


namespace max_difference_averages_l373_37396

theorem max_difference_averages (x y : ℝ) (hx : 4 ≤ x ∧ x ≤ 100) (hy : 4 ≤ y ∧ y ≤ 100) :
  ∃ (z : ℝ), z = |((x + y) / 2) - ((x + 2 * y) / 3)| ∧
  z ≤ 16 ∧
  ∃ (a b : ℝ), (4 ≤ a ∧ a ≤ 100) ∧ (4 ≤ b ∧ b ≤ 100) ∧
    |((a + b) / 2) - ((a + 2 * b) / 3)| = 16 :=
by sorry

end max_difference_averages_l373_37396


namespace count_grid_paths_l373_37335

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of distinct paths from the bottom-left corner to the top-right corner
    of an m × n grid, moving only upward or to the right, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  grid_paths m n = Nat.choose (m + n) m := by sorry

end count_grid_paths_l373_37335


namespace speed_ratio_is_four_fifths_l373_37373

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure PerpendicularMotion where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B
  d  : ℝ  -- Initial distance of B from O

/-- Equidistance condition at time t -/
def equidistant (m : PerpendicularMotion) (t : ℝ) : Prop :=
  m.vA * t = |m.d - m.vB * t|

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_is_four_fifths (m : PerpendicularMotion) :
  m.d = 600 ∧ equidistant m 3 ∧ equidistant m 12 → m.vA / m.vB = 4/5 := by
  sorry

#check speed_ratio_is_four_fifths

end speed_ratio_is_four_fifths_l373_37373


namespace ice_cream_arrangements_l373_37366

theorem ice_cream_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end ice_cream_arrangements_l373_37366


namespace power_subtraction_l373_37387

theorem power_subtraction : (2 : ℕ) ^ 4 - (2 : ℕ) ^ 3 = (2 : ℕ) ^ 3 := by
  sorry

end power_subtraction_l373_37387


namespace smallest_distance_between_complex_circles_l373_37316

theorem smallest_distance_between_complex_circles
  (z w : ℂ)
  (hz : Complex.abs (z - (2 + 2*Complex.I)) = 2)
  (hw : Complex.abs (w + (3 + 5*Complex.I)) = 4) :
  Complex.abs (z - w) ≥ Real.sqrt 74 - 6 :=
by sorry

end smallest_distance_between_complex_circles_l373_37316


namespace equation_has_two_distinct_real_roots_l373_37308

-- Define the new operation
def star_op (a b : ℝ) : ℝ := a^2 - a*b + b

-- Theorem statement
theorem equation_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star_op x₁ 3 = 5 ∧ star_op x₂ 3 = 5 :=
by sorry

end equation_has_two_distinct_real_roots_l373_37308


namespace f_x1_gt_f_x2_l373_37325

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Axioms based on the given conditions
axiom f_symmetry (x : ℝ) : f (2 - x) = f x

axiom f_derivative_condition (x : ℝ) (h : x ≠ 1) : f' x / (x - 1) < 0

axiom x1_x2_sum (x₁ x₂ : ℝ) : x₁ + x₂ > 2

axiom x1_lt_x2 (x₁ x₂ : ℝ) : x₁ < x₂

-- The theorem to be proved
theorem f_x1_gt_f_x2 (x₁ x₂ : ℝ) : f x₁ > f x₂ := by
  sorry

end f_x1_gt_f_x2_l373_37325


namespace probability_three_same_tunes_l373_37383

/-- A defective toy train that produces only two different tunes at random -/
structure DefectiveToyTrain where
  tunes : Fin 2

/-- The probability of a specific sequence of tunes occurring -/
def probability_of_sequence (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- The probability of producing n music tunes of the same type in a row -/
def probability_same_tune (n : ℕ) : ℚ :=
  2 * probability_of_sequence n

theorem probability_three_same_tunes :
  probability_same_tune 3 = 1 / 4 := by sorry

end probability_three_same_tunes_l373_37383


namespace last_four_digits_of_5_pow_2015_l373_37375

theorem last_four_digits_of_5_pow_2015 : ∃ n : ℕ, 5^2015 ≡ 8125 [MOD 10000] := by
  sorry

end last_four_digits_of_5_pow_2015_l373_37375


namespace triangle_calculation_l373_37379

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b + 2 * a

-- State the theorem
theorem triangle_calculation : triangle (-3) (triangle (-4) (1/2)) = 24 := by
  sorry

end triangle_calculation_l373_37379


namespace consecutive_integers_sum_l373_37368

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = -336 ∧ n < 0 → (n - 1) + n + (n + 1) = -21 := by
  sorry

end consecutive_integers_sum_l373_37368


namespace folded_perimeter_not_greater_l373_37381

/-- Represents a polygon in 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculates the perimeter of a polygon -/
def perimeter (p : Polygon) : ℝ := sorry

/-- Folds a polygon along a line and glues the halves together -/
def fold_and_glue (p : Polygon) (l : Line) : Polygon := sorry

/-- Theorem: The perimeter of a folded and glued polygon is not greater than the original -/
theorem folded_perimeter_not_greater (p : Polygon) (l : Line) :
  perimeter (fold_and_glue p l) ≤ perimeter p := by sorry

end folded_perimeter_not_greater_l373_37381


namespace sequence_periodicity_l373_37370

def is_periodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∃ k : ℕ, ∀ n ≥ k, a n = a (n + p)

def smallest_period (a : ℕ → ℝ) (p : ℕ) : Prop :=
  is_periodic a p ∧ ∀ q < p, ¬ is_periodic a q

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h1 : ∃ n, a n ≠ 0)
  (h2 : ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n) :
  smallest_period a 9 :=
sorry

end sequence_periodicity_l373_37370


namespace valid_factorization_l373_37361

theorem valid_factorization (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

#check valid_factorization

end valid_factorization_l373_37361


namespace smallest_n_for_sqrt_difference_l373_37307

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ+, (∀ m : ℕ+, m < n → Real.sqrt m.val - Real.sqrt (m.val - 1) ≥ 0.1) ∧ (Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.1) ∧ n = 26 := by
  sorry

end smallest_n_for_sqrt_difference_l373_37307


namespace solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l373_37329

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 3x + 4
theorem solution_set_f_leq_3x_plus_4 :
  {x : ℝ | f x ≤ 3 * x + 4} = {x : ℝ | x ≥ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_all_reals (m : ℝ) :
  ({x : ℝ | f x ≥ m} = Set.univ) ↔ m ∈ Set.Iic 4 :=
sorry

#check solution_set_f_leq_3x_plus_4
#check range_of_m_for_f_geq_m_all_reals

end solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l373_37329


namespace ezekiel_hike_third_day_l373_37315

/-- Represents a three-day hike --/
structure ThreeDayHike where
  total_distance : ℕ
  day1_distance : ℕ
  day2_distance : ℕ

/-- Calculates the distance covered on the third day of a three-day hike --/
def third_day_distance (hike : ThreeDayHike) : ℕ :=
  hike.total_distance - (hike.day1_distance + hike.day2_distance)

/-- Theorem stating that for the given hike parameters, the third day distance is 22 km --/
theorem ezekiel_hike_third_day :
  let hike : ThreeDayHike := {
    total_distance := 50,
    day1_distance := 10,
    day2_distance := 18
  }
  third_day_distance hike = 22 := by
  sorry


end ezekiel_hike_third_day_l373_37315


namespace unique_valid_ticket_l373_37382

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def is_even (n : ℕ) : Prop := n % 2 = 0

def ticket_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_prime (n % 10) ∧
  is_multiple_of_5 ((n / 10) % 10) ∧
  is_even ((n / 100) % 10) ∧
  n / 1000 = 3 * (n % 10)

theorem unique_valid_ticket : ∀ n : ℕ, ticket_valid n ↔ n = 9853 :=
sorry

end unique_valid_ticket_l373_37382


namespace first_agency_mile_rate_calculation_l373_37391

-- Define the constants
def first_agency_daily_rate : ℝ := 20.25
def second_agency_daily_rate : ℝ := 18.25
def second_agency_mile_rate : ℝ := 0.22
def crossover_miles : ℝ := 25.0

-- Define the theorem
theorem first_agency_mile_rate_calculation :
  ∃ (x : ℝ),
    first_agency_daily_rate + crossover_miles * x =
    second_agency_daily_rate + crossover_miles * second_agency_mile_rate ∧
    x = 0.14 := by
  sorry

end first_agency_mile_rate_calculation_l373_37391


namespace conference_handshakes_l373_37349

/-- The number of unique handshakes in a circular seating arrangement --/
def unique_handshakes (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with 30 people, 
    where each person shakes hands only with their immediate neighbors, 
    the number of unique handshakes is equal to 30. --/
theorem conference_handshakes : 
  unique_handshakes 30 = 30 := by
  sorry

end conference_handshakes_l373_37349


namespace intersection_of_A_and_B_l373_37360

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l373_37360


namespace fraction_equation_solution_l373_37394

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2) ∧ x = 1 / 3 := by
  sorry

end fraction_equation_solution_l373_37394


namespace units_digit_17_squared_times_29_l373_37342

theorem units_digit_17_squared_times_29 : (17^2 * 29) % 10 = 1 := by
  sorry

end units_digit_17_squared_times_29_l373_37342


namespace solve_dimes_problem_l373_37386

def dimes_problem (initial_dimes : ℕ) (given_to_mother : ℕ) (final_dimes : ℕ) : Prop :=
  ∃ (dimes_from_dad : ℕ),
    initial_dimes - given_to_mother + dimes_from_dad = final_dimes

theorem solve_dimes_problem :
  dimes_problem 7 4 11 → ∃ (dimes_from_dad : ℕ), dimes_from_dad = 8 :=
by
  sorry

end solve_dimes_problem_l373_37386


namespace jan_extra_distance_l373_37343

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_time : ℝ
  han_speed : ℝ
  jan_time : ℝ
  jan_speed : ℝ
  han_extra_distance : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_time = s.ian_time + 2 ∧
  s.han_speed = s.ian_speed + 10 ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_extra_distance = 120

/-- The theorem stating that Jan drove 195 miles more than Ian -/
theorem jan_extra_distance (s : DrivingScenario) 
  (h : scenario_conditions s) : 
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 195 :=
sorry


end jan_extra_distance_l373_37343


namespace liams_numbers_l373_37384

theorem liams_numbers (x y : ℤ) : 
  (3 * x + 2 * y = 75) →  -- Sum of five numbers is 75
  (x = 15) →              -- The number written three times is 15
  (x * y % 5 = 0) →       -- Product of the two numbers is a multiple of 5
  (y = 15) :=             -- The other number (written twice) is 15
by sorry

end liams_numbers_l373_37384


namespace angle2_value_l373_37314

-- Define the angles
variable (angle1 angle2 angle3 : ℝ)

-- Define the conditions
def complementary (a b : ℝ) : Prop := a + b = 90
def supplementary (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle2_value (h1 : complementary angle1 angle2)
                     (h2 : supplementary angle1 angle3)
                     (h3 : angle3 = 125) :
  angle2 = 35 := by
  sorry

end angle2_value_l373_37314


namespace simplified_expression_l373_37398

theorem simplified_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  (108 * (Real.sqrt 10 + Real.sqrt 14 - Real.sqrt 6 - Real.sqrt 490)) / (-59) :=
by sorry

end simplified_expression_l373_37398


namespace max_y_over_x_l373_37305

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 3 + 2 * Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ max :=
by sorry

end max_y_over_x_l373_37305


namespace abs_eq_sqrt_square_domain_eq_reals_l373_37318

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem domain_eq_reals : Set.range (fun x => |x|) = Set.range (fun x => Real.sqrt (x^2)) := by sorry

end abs_eq_sqrt_square_domain_eq_reals_l373_37318


namespace brenda_mice_problem_l373_37333

theorem brenda_mice_problem (total_mice : ℕ) : 
  (∃ (given_to_robbie sold_to_store sold_as_feeder remaining : ℕ),
    given_to_robbie = total_mice / 6 ∧
    sold_to_store = 3 * given_to_robbie ∧
    sold_as_feeder = (total_mice - given_to_robbie - sold_to_store) / 2 ∧
    remaining = total_mice - given_to_robbie - sold_to_store - sold_as_feeder ∧
    remaining = 4 ∧
    total_mice % 3 = 0) →
  total_mice / 3 = 8 := by
sorry

end brenda_mice_problem_l373_37333


namespace hyperbola_equation_l373_37332

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote of C
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse that shares a focus with C
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y, asymptote x y → hyperbola a b x y) ∧
  (∃ x y, ellipse x y ∧ hyperbola a b x y) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end hyperbola_equation_l373_37332


namespace convex_quadrilateral_probability_l373_37304

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of convex quadrilaterals that can be formed -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end convex_quadrilateral_probability_l373_37304


namespace unique_solution_iff_a_in_open_interval_l373_37340

/-- The system of equations has exactly one solution if and only if 0 < a < 4 -/
theorem unique_solution_iff_a_in_open_interval (a : ℝ) :
  (∃! x y z : ℝ, x + y + z = 0 ∧ x*y + y*z + a*z*x = 0) ↔ 0 < a ∧ a < 4 :=
sorry

end unique_solution_iff_a_in_open_interval_l373_37340


namespace quadratic_expression_value_l373_37363

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 := by
  sorry

end quadratic_expression_value_l373_37363
