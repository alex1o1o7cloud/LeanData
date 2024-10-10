import Mathlib

namespace interest_difference_l1146_114618

def principal : ℚ := 250
def rate : ℚ := 4
def time : ℚ := 8

def simple_interest (p r t : ℚ) : ℚ := (p * r * t) / 100

theorem interest_difference :
  principal - simple_interest principal rate time = 170 := by
  sorry

end interest_difference_l1146_114618


namespace right_triangle_area_l1146_114666

/-- The area of a right triangle formed by two perpendicular vectors -/
theorem right_triangle_area (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  let area := (1/2) * abs (a.1 * b.2 - a.2 * b.1)
  (a = (3, 4) ∧ b = (-4, 3)) → area = 12.5 := by
sorry

end right_triangle_area_l1146_114666


namespace greendale_points_are_130_l1146_114600

/-- Calculates the total points for Greendale High School in a basketball tournament -/
def greendalePoints : ℕ :=
  let rooseveltFirstGame : ℕ := 30
  let rooseveltSecondGame : ℕ := rooseveltFirstGame / 2
  let rooseveltThirdGame : ℕ := rooseveltSecondGame * 3
  let rooseveltTotalBeforeBonus : ℕ := rooseveltFirstGame + rooseveltSecondGame + rooseveltThirdGame
  let rooseveltBonus : ℕ := 50
  let rooseveltTotal : ℕ := rooseveltTotalBeforeBonus + rooseveltBonus
  let pointDifference : ℕ := 10
  rooseveltTotal - pointDifference

/-- Theorem stating that Greendale High School's total points are 130 -/
theorem greendale_points_are_130 : greendalePoints = 130 := by
  sorry

end greendale_points_are_130_l1146_114600


namespace power_log_fourth_root_l1146_114662

theorem power_log_fourth_root (x : ℝ) (h : x > 0) :
  ((625 ^ (Real.log x / Real.log 5)) ^ (1/4) : ℝ) = x :=
by sorry

end power_log_fourth_root_l1146_114662


namespace car_trip_duration_l1146_114619

theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 6 →
  additional_speed = 46 →
  average_speed = 34 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 := by
  sorry

end car_trip_duration_l1146_114619


namespace modulus_of_complex_fraction_l1146_114625

/-- The modulus of the complex number z = (1+3i)/(1-i) is equal to √5 -/
theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_complex_fraction_l1146_114625


namespace fish_population_approximation_l1146_114601

/-- Represents the fish population in an ocean reserve --/
structure FishPopulation where
  initialPopulation : ℕ
  taggedFish : ℕ
  secondCatchSize : ℕ
  taggedInSecondCatch : ℕ
  monthlyMigration : ℕ
  monthlyDeaths : ℕ
  months : ℕ

/-- Calculates the approximate number of fish in the ocean reserve after a given number of months --/
def approximateFishPopulation (fp : FishPopulation) : ℕ :=
  let totalChange := fp.months * (fp.monthlyMigration - fp.monthlyDeaths)
  let finalPopulation := (fp.secondCatchSize * fp.taggedFish) / fp.taggedInSecondCatch + totalChange
  finalPopulation

/-- Theorem stating that the approximate number of fish in the ocean reserve after three months is 71429 --/
theorem fish_population_approximation (fp : FishPopulation) 
  (h1 : fp.taggedFish = 1000)
  (h2 : fp.secondCatchSize = 500)
  (h3 : fp.taggedInSecondCatch = 7)
  (h4 : fp.monthlyMigration = 150)
  (h5 : fp.monthlyDeaths = 200)
  (h6 : fp.months = 3) :
  approximateFishPopulation fp = 71429 := by
  sorry

#eval approximateFishPopulation {
  initialPopulation := 0,  -- Not used in the calculation
  taggedFish := 1000,
  secondCatchSize := 500,
  taggedInSecondCatch := 7,
  monthlyMigration := 150,
  monthlyDeaths := 200,
  months := 3
}

end fish_population_approximation_l1146_114601


namespace absolute_difference_inequality_l1146_114644

theorem absolute_difference_inequality (x : ℝ) : 
  |x - 1| - |x - 2| > (1/2) ↔ x > (7/4) := by sorry

end absolute_difference_inequality_l1146_114644


namespace corner_sum_is_ten_l1146_114626

/-- Represents a Go board as a function from coordinates to real numbers -/
def GoBoard : Type := Fin 18 → Fin 18 → ℝ

/-- The property that any 2x2 square on the board sums to 10 -/
def valid_board (board : GoBoard) : Prop :=
  ∀ i j, i < 17 → j < 17 →
    board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1) = 10

/-- The sum of the four corner squares -/
def corner_sum (board : GoBoard) : ℝ :=
  board 0 0 + board 0 17 + board 17 0 + board 17 17

/-- Theorem: For any valid Go board, the sum of the four corners is 10 -/
theorem corner_sum_is_ten (board : GoBoard) (h : valid_board board) :
  corner_sum board = 10 := by
  sorry

end corner_sum_is_ten_l1146_114626


namespace equation_solutions_count_l1146_114658

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), (∀ x ∈ s, 9 * x^2 - 63 * ⌊x⌋ + 72 = 0) ∧ s.card = 2 :=
sorry

end equation_solutions_count_l1146_114658


namespace imaginary_part_of_3_minus_4i_l1146_114671

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end imaginary_part_of_3_minus_4i_l1146_114671


namespace clothing_price_problem_l1146_114694

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) 
  (h1 : total_spent = 610)
  (h2 : num_pieces = 7)
  (h3 : price1 = 49)
  (h4 : price2 = 81)
  : (total_spent - price1 - price2) / (num_pieces - 2) = 96 := by
  sorry

end clothing_price_problem_l1146_114694


namespace janet_freelance_earnings_l1146_114688

/-- Calculates the difference in monthly earnings between freelancing and current job --/
def freelance_earnings_difference (
  hours_per_week : ℕ)
  (current_wage : ℚ)
  (freelance_wage : ℚ)
  (weeks_per_month : ℕ)
  (extra_fica_per_week : ℚ)
  (healthcare_premium_per_month : ℚ) : ℚ :=
  let wage_difference := freelance_wage - current_wage
  let weekly_earnings_difference := wage_difference * hours_per_week
  let monthly_earnings_difference := weekly_earnings_difference * weeks_per_month
  let extra_monthly_expenses := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  monthly_earnings_difference - extra_monthly_expenses

/-- Theorem stating the earnings difference for Janet's specific situation --/
theorem janet_freelance_earnings :
  freelance_earnings_difference 40 30 40 4 25 400 = 1100 := by
  sorry

end janet_freelance_earnings_l1146_114688


namespace zeros_of_cosine_minus_one_l1146_114695

theorem zeros_of_cosine_minus_one (ω : ℝ) : 
  (ω > 0) →
  (∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₂ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₃ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₁ ≠ x₂) ∧ (x₂ ≠ x₃) ∧ (x₁ ≠ x₃) ∧
    (Real.cos (ω * x₁) = 1) ∧ 
    (Real.cos (ω * x₂) = 1) ∧ 
    (Real.cos (ω * x₃) = 1) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (ω * x) = 1 → (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (2 ≤ ω ∧ ω < 3) :=
by sorry

end zeros_of_cosine_minus_one_l1146_114695


namespace parabola_translation_l1146_114612

def original_parabola (x : ℝ) : ℝ := x^2 + 1

def transformed_parabola (x : ℝ) : ℝ := x^2 + 4*x + 5

def translation_distance : ℝ := 2

theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola (x + translation_distance) = original_parabola x :=
by sorry

end parabola_translation_l1146_114612


namespace billy_reads_three_books_l1146_114654

/-- Represents Billy's reading activity over the weekend --/
structure BillyReading where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  time_available : ℝ  -- Total time available for reading in hours
  book_pages : ℕ  -- Number of pages in each book
  speed_decrease : ℝ  -- Percentage decrease in reading speed after each book

/-- Calculates the number of books Billy can read --/
def books_read (b : BillyReading) : ℕ :=
  sorry

/-- Theorem stating that Billy can read exactly 3 books --/
theorem billy_reads_three_books :
  let b : BillyReading := {
    initial_speed := 60,
    time_available := 16 * 0.35,
    book_pages := 80,
    speed_decrease := 0.1
  }
  books_read b = 3 := by sorry

end billy_reads_three_books_l1146_114654


namespace cylinder_volume_l1146_114681

/-- The volume of a cylinder with base diameter and height both equal to 3 is (27/4)π. -/
theorem cylinder_volume (d h : ℝ) (hd : d = 3) (hh : h = 3) :
  let r := d / 2
  π * r^2 * h = (27 / 4) * π :=
by sorry

end cylinder_volume_l1146_114681


namespace isosceles_triangle_height_l1146_114637

/-- Given an isosceles triangle and a rectangle with the same area, where the base of the triangle
    equals the width of the rectangle (10 units), and the length of the rectangle is twice its width,
    prove that the height of the triangle is 40 units. -/
theorem isosceles_triangle_height (triangle_area rectangle_area : ℝ) 
  (triangle_base rectangle_width rectangle_length : ℝ) (triangle_height : ℝ) : 
  triangle_area = rectangle_area →
  triangle_base = rectangle_width →
  triangle_base = 10 →
  rectangle_length = 2 * rectangle_width →
  triangle_area = 1/2 * triangle_base * triangle_height →
  rectangle_area = rectangle_width * rectangle_length →
  triangle_height = 40 := by
  sorry

end isosceles_triangle_height_l1146_114637


namespace paper_I_maximum_mark_l1146_114696

theorem paper_I_maximum_mark :
  ∃ (M : ℕ),
    (M : ℚ) * (55 : ℚ) / (100 : ℚ) = (65 : ℚ) + (35 : ℚ) ∧
    M = 182 := by
  sorry

end paper_I_maximum_mark_l1146_114696


namespace no_k_satisfies_condition_l1146_114647

-- Define a function to get the nth odd prime number
def nthOddPrime (n : ℕ) : ℕ := sorry

-- Define a function to calculate the product of the first k odd primes
def productOfFirstKOddPrimes (k : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect power greater than 1
def isPerfectPowerGreaterThanOne (n : ℕ) : Prop := sorry

-- Theorem statement
theorem no_k_satisfies_condition :
  ∀ k : ℕ, k > 0 → ¬(isPerfectPowerGreaterThanOne (productOfFirstKOddPrimes k - 1)) := by
  sorry

end no_k_satisfies_condition_l1146_114647


namespace set_operations_l1146_114640

-- Define the sets A and B
def A : Set ℝ := {x | x = 0 ∨ ∃ y, x = |y|}
def B : Set ℝ := {-1, 0, 1}

-- State the theorem
theorem set_operations (h : A ⊆ B) :
  (A ∩ B = {0, 1}) ∧
  (A ∪ B = {-1, 0, 1}) ∧
  (B \ A = {-1}) := by
  sorry

end set_operations_l1146_114640


namespace complex_conjugate_roots_imply_zero_coefficients_l1146_114634

/-- Given a quadratic equation z^2 + (6 + pi)z + (10 + qi) = 0 where p and q are real numbers,
    if the roots are complex conjugates, then p = 0 and q = 0 -/
theorem complex_conjugate_roots_imply_zero_coefficients (p q : ℝ) :
  (∃ x y : ℝ, (Complex.I : ℂ)^2 = -1 ∧
    (x + y * Complex.I) * (x - y * Complex.I) = -(6 + p * Complex.I) * (x + y * Complex.I) - (10 + q * Complex.I)) →
  p = 0 ∧ q = 0 := by
sorry

end complex_conjugate_roots_imply_zero_coefficients_l1146_114634


namespace circle_area_ratio_l1146_114639

/-- If a 60° arc on circle A has the same length as a 40° arc on circle B,
    then the ratio of the area of circle A to the area of circle B is 4/9 -/
theorem circle_area_ratio (r_A r_B : ℝ) (h : r_A > 0 ∧ r_B > 0) :
  (60 / 360) * (2 * Real.pi * r_A) = (40 / 360) * (2 * Real.pi * r_B) →
  (Real.pi * r_A ^ 2) / (Real.pi * r_B ^ 2) = 4 / 9 := by
sorry


end circle_area_ratio_l1146_114639


namespace max_a_value_l1146_114633

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

/-- Predicate to check if a point satisfies the line equation -/
def SatisfiesEquation (m : ℚ) (x y : ℤ) : Prop :=
  LineEquation m x = y

/-- The main theorem -/
theorem max_a_value : 
  ∃ (a : ℚ), a = 101 / 151 ∧ 
  (∀ (m : ℚ), 2/3 < m → m < a → 
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬SatisfiesEquation m x y) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m : ℚ), 2/3 < m ∧ m < a' ∧
      ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ SatisfiesEquation m x y) :=
sorry

end max_a_value_l1146_114633


namespace ellipse_eccentricity_min_mn_l1146_114697

/-- Given two positive real numbers m and n satisfying 1/m + 2/n = 1,
    the eccentricity of the ellipse x²/m² + y²/n² = 1 is √3/2
    when mn takes its minimum value. -/
theorem ellipse_eccentricity_min_mn (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h : 1/m + 2/n = 1) :
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (min_mn : ℝ), (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1/m' + 2/n' = 1 → m' * n' ≥ min_mn) ∧
    (m * n = min_mn → e = Real.sqrt 3 / 2) :=
sorry

end ellipse_eccentricity_min_mn_l1146_114697


namespace max_value_2a_minus_b_l1146_114656

theorem max_value_2a_minus_b :
  ∃ (M : ℝ), M = 2 + Real.sqrt 5 ∧
  (∀ a b : ℝ, a^2 + b^2 - 2*a = 0 → 2*a - b ≤ M) ∧
  (∃ a b : ℝ, a^2 + b^2 - 2*a = 0 ∧ 2*a - b = M) :=
by sorry

end max_value_2a_minus_b_l1146_114656


namespace min_perimeter_isosceles_triangles_l1146_114690

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Check if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculate the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculate the area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 2

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    10 * t1.base = 9 * t2.base ∧
    perimeter t1 = 362 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      10 * s1.base = 9 * s2.base →
      perimeter s1 ≥ 362) :=
sorry

end min_perimeter_isosceles_triangles_l1146_114690


namespace sound_distance_at_18C_l1146_114687

/-- Represents the speed of sound in air as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 331 + 0.6 * t

/-- Calculates the distance traveled by sound given time and temperature -/
def distance_traveled (time : ℝ) (temp : ℝ) : ℝ :=
  (speed_of_sound temp) * time

/-- Theorem: The distance traveled by sound in 5 seconds at 18°C is approximately 1709 meters -/
theorem sound_distance_at_18C : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_traveled 5 18 - 1709| < ε :=
sorry

end sound_distance_at_18C_l1146_114687


namespace solar_project_profit_l1146_114620

/-- Represents the net profit of a solar power generation project -/
def net_profit (n : ℕ+) : ℤ :=
  n - (4 * n^2 + 20 * n) - 144

/-- Theorem stating the net profit expression and when the project starts to make profit -/
theorem solar_project_profit :
  (∀ n : ℕ+, net_profit n = -4 * n^2 + 80 * n - 144) ∧
  (∀ n : ℕ+, net_profit n > 0 ↔ n ≥ 3) := by
  sorry

end solar_project_profit_l1146_114620


namespace line_equation_proof_l1146_114680

/-- Proves that the equation of a line with slope -2 and y-intercept 3 is 2x + y - 3 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  let slope : ℝ := -2
  let y_intercept : ℝ := 3
  let line_equation := fun (x y : ℝ) => 2 * x + y - 3 = 0
  line_equation x y ↔ y = slope * x + y_intercept :=
by sorry

end line_equation_proof_l1146_114680


namespace two_rooks_placement_count_l1146_114621

/-- The size of a standard chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on a chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares attacked by a rook (excluding its own square) -/
def attackedSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks of different colors on a chessboard
    such that they do not attack each other -/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - attackedSquares)

theorem two_rooks_placement_count :
  twoRooksPlacement = 3136 := by sorry

end two_rooks_placement_count_l1146_114621


namespace equation_solution_l1146_114669

theorem equation_solution (x : ℝ) :
  (∃ (n : ℤ), x = π / 18 + 2 * π * n / 9) ∨ (∃ (s : ℤ), x = 2 * π * s / 3) ↔
  (((1 - (Real.cos (15 * x))^7 * (Real.cos (9 * x))^2)^(1/4) = Real.sin (9 * x)) ∧
   Real.sin (9 * x) ≥ 0) := by
  sorry

end equation_solution_l1146_114669


namespace min_value_theorem_l1146_114643

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ (3 * x + 4 * y = 5 ↔ x = 1 ∧ y = 1/2) :=
by sorry

end min_value_theorem_l1146_114643


namespace experiment_A_not_control_based_l1146_114648

-- Define the type for experiments
inductive Experiment
| A
| B
| C
| D

-- Define a predicate for experiments designed based on the principle of control
def is_control_based (e : Experiment) : Prop :=
  match e with
  | Experiment.A => False
  | _ => True

-- Theorem statement
theorem experiment_A_not_control_based :
  is_control_based Experiment.B ∧
  is_control_based Experiment.C ∧
  is_control_based Experiment.D →
  ¬is_control_based Experiment.A :=
by
  sorry

end experiment_A_not_control_based_l1146_114648


namespace floor_product_equality_l1146_114655

theorem floor_product_equality (x : ℝ) : ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
  sorry

end floor_product_equality_l1146_114655


namespace percentage_calculation_l1146_114663

theorem percentage_calculation (part whole : ℝ) (h1 : part = 375.2) (h2 : whole = 12546.8) :
  (part / whole) * 100 = 2.99 := by
  sorry

end percentage_calculation_l1146_114663


namespace daisy_tuesday_toys_l1146_114668

/-- The number of dog toys Daisy had on various days --/
structure DaisyToys where
  monday : ℕ
  tuesday_before : ℕ
  tuesday_after : ℕ
  wednesday_new : ℕ
  total_if_found : ℕ

/-- Theorem stating the number of toys Daisy had on Tuesday before new purchases --/
theorem daisy_tuesday_toys (d : DaisyToys)
  (h1 : d.monday = 5)
  (h2 : d.tuesday_after = d.tuesday_before + 3)
  (h3 : d.wednesday_new = 5)
  (h4 : d.total_if_found = 13)
  (h5 : d.total_if_found = d.tuesday_before + 3 + d.wednesday_new) :
  d.tuesday_before = 5 := by
  sorry

#check daisy_tuesday_toys

end daisy_tuesday_toys_l1146_114668


namespace min_yellow_fraction_l1146_114659

/-- Represents a cube with its edge length and number of blue and yellow subcubes. -/
structure Cube where
  edge_length : ℕ
  blue_cubes : ℕ
  yellow_cubes : ℕ

/-- Calculates the minimum yellow surface area for a given cube configuration. -/
def min_yellow_surface_area (c : Cube) : ℕ :=
  sorry

/-- Calculates the total surface area of a cube. -/
def total_surface_area (c : Cube) : ℕ :=
  6 * c.edge_length * c.edge_length

/-- The main theorem stating the minimum fraction of yellow surface area. -/
theorem min_yellow_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.blue_cubes = 48)
  (h3 : c.yellow_cubes = 16)
  (h4 : c.blue_cubes + c.yellow_cubes = c.edge_length * c.edge_length * c.edge_length) :
  min_yellow_surface_area c / total_surface_area c = 1 / 12 :=
sorry

end min_yellow_fraction_l1146_114659


namespace tangent_line_and_perpendicular_points_l1146_114642

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points :
  -- Part 1: Equation of tangent line at (1, -1)
  (∀ x y : ℝ, (x = 1 ∧ y = f 1) → (2*x - y - 3 = 0)) ∧
  -- Part 2: Points where tangent is perpendicular to y = -1/2x + 3
  (∀ x : ℝ, (f' x = 2) → (x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, (x = 1 ∨ x = -1) → f x = -1) :=
by sorry

end tangent_line_and_perpendicular_points_l1146_114642


namespace octahedron_tetrahedron_volume_relation_l1146_114616

/-- Regular octahedron with side length 2√2 -/
def octahedron : Real → Set (Fin 3 → ℝ) := sorry

/-- Tetrahedron with vertices at the centers of octahedron faces -/
def tetrahedron (O : Set (Fin 3 → ℝ)) : Set (Fin 3 → ℝ) := sorry

/-- Volume of a set in ℝ³ -/
def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

theorem octahedron_tetrahedron_volume_relation :
  let O := octahedron (2 * Real.sqrt 2)
  let T := tetrahedron O
  volume O = 4 * volume T →
  volume T = (4 * Real.sqrt 2) / 3 := by sorry

end octahedron_tetrahedron_volume_relation_l1146_114616


namespace leaders_photo_theorem_l1146_114630

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k objects from n distinct objects and arrange them. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then (permutations n) / (permutations (n - k)) else 0

/-- The number of arrangements for the leaders' photo. -/
def leaders_photo_arrangements : ℕ := 
  (arrangements 2 1) * (arrangements 18 18)

theorem leaders_photo_theorem : 
  leaders_photo_arrangements = (arrangements 2 1) * (arrangements 18 18) := by
  sorry

end leaders_photo_theorem_l1146_114630


namespace at_least_one_not_less_than_two_l1146_114638

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l1146_114638


namespace largest_angle_in_pentagon_l1146_114610

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 60 →
  B = 85 →
  C = D →
  E = 2 * C + 15 →
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 205 :=
by sorry

end largest_angle_in_pentagon_l1146_114610


namespace min_red_chips_is_76_l1146_114613

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies the given conditions -/
def isValidChipCount (c : ChipCount) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 75

/-- The minimum number of red chips that satisfies the conditions -/
def minRedChips : ℕ := 76

/-- Theorem stating that the minimum number of red chips is 76 -/
theorem min_red_chips_is_76 :
  ∀ c : ChipCount, isValidChipCount c → c.red ≥ minRedChips :=
by sorry

end min_red_chips_is_76_l1146_114613


namespace inverse_of_inverse_nine_l1146_114684

def f (x : ℝ) : ℝ := 5 * x + 7

theorem inverse_of_inverse_nine :
  let f_inv (x : ℝ) := (x - 7) / 5
  f_inv (f_inv 9) = -33 / 25 := by
sorry

end inverse_of_inverse_nine_l1146_114684


namespace edge_length_of_total_72_l1146_114674

/-- Represents a rectangular prism with equal edge lengths -/
structure EqualEdgePrism where
  edge_length : ℝ
  total_length : ℝ
  total_length_eq : total_length = 12 * edge_length

/-- Theorem: If the sum of all edge lengths in an equal edge prism is 72 cm, 
    then the length of one edge is 6 cm -/
theorem edge_length_of_total_72 (prism : EqualEdgePrism) 
  (h : prism.total_length = 72) : prism.edge_length = 6 := by
  sorry

end edge_length_of_total_72_l1146_114674


namespace solution_set_is_open_interval_l1146_114661

def solution_set : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (-1 : ℝ) 1 := by sorry

end solution_set_is_open_interval_l1146_114661


namespace sine_cosine_problem_l1146_114641

theorem sine_cosine_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < π/2) 
  (h2 : Real.sin x + Real.cos x = -1/5) : 
  (Real.sin x - Real.cos x = 7/5) ∧ 
  ((Real.sin (π + x) + Real.sin (3*π/2 - x)) / (Real.tan (π - x) + Real.sin (π/2 - x)) = 3/11) := by
  sorry

end sine_cosine_problem_l1146_114641


namespace last_two_digits_sum_l1146_114605

theorem last_two_digits_sum (n : ℕ) : n = 25 → (6^n + 14^n) % 100 = 0 := by
  sorry

end last_two_digits_sum_l1146_114605


namespace equal_intercept_line_equation_l1146_114628

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, 3) -/
  passes_through_point : m * 2 + b = 3
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of the line is either x + y - 5 = 0 or 3x - 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.m * x + l.b → x + y = 5) ∨
  (∀ x y, y = l.m * x + l.b → 3*x - 2*y = 0) :=
sorry

end equal_intercept_line_equation_l1146_114628


namespace musicians_performing_l1146_114623

/-- Represents a musical group --/
inductive MusicalGroup
| Quartet
| Trio
| Duet

/-- The number of musicians in each type of group --/
def group_size (g : MusicalGroup) : ℕ :=
  match g with
  | MusicalGroup.Quartet => 4
  | MusicalGroup.Trio => 3
  | MusicalGroup.Duet => 2

/-- The original schedule of performances --/
def original_schedule : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 4), (MusicalGroup.Duet, 5), (MusicalGroup.Trio, 6)]

/-- The changes to the schedule --/
def schedule_changes : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 1), (MusicalGroup.Duet, 2), (MusicalGroup.Trio, 1)]

/-- Calculate the total number of musicians given a schedule --/
def total_musicians (schedule : List (MusicalGroup × ℕ)) : ℕ :=
  schedule.foldl (fun acc (g, n) => acc + n * group_size g) 0

/-- The main theorem --/
theorem musicians_performing (
  orig_schedule : List (MusicalGroup × ℕ)) 
  (changes : List (MusicalGroup × ℕ)) :
  orig_schedule = original_schedule →
  changes = schedule_changes →
  total_musicians orig_schedule - 
  (total_musicians changes + 1) = 35 := by
  sorry

end musicians_performing_l1146_114623


namespace lines_parallel_iff_a_eq_3_l1146_114649

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

-- Define parallel lines
def parallel (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ → line2 a x₂ y₂ → 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) → (a * (x₂ - x₁) = 2 * (y₁ - y₂) ∧ 3 * (x₂ - x₁) = (a - 1) * (y₁ - y₂))

-- Theorem statement
theorem lines_parallel_iff_a_eq_3 : ∀ a : ℝ, parallel a ↔ a = 3 := by sorry

end lines_parallel_iff_a_eq_3_l1146_114649


namespace repeating_decimal_sum_l1146_114646

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 = (8 : ℚ) / 33 := by sorry

end repeating_decimal_sum_l1146_114646


namespace football_player_average_increase_l1146_114611

theorem football_player_average_increase (goals_fifth_match : ℕ) (total_goals : ℕ) :
  goals_fifth_match = 2 →
  total_goals = 8 →
  (total_goals / 5 : ℚ) - ((total_goals - goals_fifth_match) / 4 : ℚ) = 1/10 := by
  sorry

end football_player_average_increase_l1146_114611


namespace problem_statement_l1146_114675

theorem problem_statement :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ∧
  (∀ x : ℝ, 0 < x → x < π / 2 → x > Real.sin x) :=
by sorry

end problem_statement_l1146_114675


namespace max_snacks_is_11_l1146_114692

/-- Represents the number of snacks in a pack -/
inductive SnackPack
  | Single : SnackPack
  | Pack4 : SnackPack
  | Pack7 : SnackPack

/-- The cost of a snack pack in dollars -/
def cost : SnackPack → ℕ
  | SnackPack.Single => 2
  | SnackPack.Pack4 => 6
  | SnackPack.Pack7 => 9

/-- The number of snacks in a pack -/
def snacks : SnackPack → ℕ
  | SnackPack.Single => 1
  | SnackPack.Pack4 => 4
  | SnackPack.Pack7 => 7

/-- The budget in dollars -/
def budget : ℕ := 15

/-- A purchase is a list of snack packs -/
def Purchase := List SnackPack

/-- The total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + cost pack) 0

/-- The total number of snacks in a purchase -/
def totalSnacks (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + snacks pack) 0

/-- A purchase is valid if its total cost is within the budget -/
def isValidPurchase (p : Purchase) : Prop :=
  totalCost p ≤ budget

/-- The theorem stating that 11 is the maximum number of snacks that can be purchased -/
theorem max_snacks_is_11 :
  ∀ p : Purchase, isValidPurchase p → totalSnacks p ≤ 11 :=
sorry

end max_snacks_is_11_l1146_114692


namespace part_to_third_ratio_l1146_114664

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end part_to_third_ratio_l1146_114664


namespace treehouse_planks_l1146_114603

theorem treehouse_planks (initial_planks : ℕ) (total_planks : ℕ) (planks_from_forest : ℕ) :
  initial_planks = 15 →
  total_planks = 35 →
  total_planks = initial_planks + 2 * planks_from_forest →
  planks_from_forest = 10 := by
  sorry

end treehouse_planks_l1146_114603


namespace fraction_equality_l1146_114667

theorem fraction_equality (x y z : ℝ) (h1 : x / 2 = y / 3) (h2 : x / 2 = z / 5) (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  sorry

end fraction_equality_l1146_114667


namespace second_number_value_l1146_114657

theorem second_number_value (x : ℝ) : 3 + x * (8 - 3) = 24.16 → x = 4.232 := by
  sorry

end second_number_value_l1146_114657


namespace office_network_connections_l1146_114682

/-- A network of switches where each switch connects to exactly four others. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  connection_count : ℕ

/-- The theorem stating the correct number of connections in the given network. -/
theorem office_network_connections (network : SwitchNetwork)
  (h1 : network.num_switches = 30)
  (h2 : network.connections_per_switch = 4) :
  network.connection_count = 60 := by
  sorry

end office_network_connections_l1146_114682


namespace range_of_x_minus_y_l1146_114679

theorem range_of_x_minus_y :
  ∀ x y : ℝ, 2 < x ∧ x < 4 → -1 < y ∧ y < 3 →
  ∃ z : ℝ, -1 < z ∧ z < 5 ∧ z = x - y ∧
  ∀ w : ℝ, w = x - y → -1 < w ∧ w < 5 :=
by sorry

end range_of_x_minus_y_l1146_114679


namespace midpoint_property_l1146_114604

/-- Given two points A and B in the plane, if C is their midpoint,
    then 2x - 4y = 0, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (hA : A = (20, 10)) (hB : B = (10, 5)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = 0 := by
sorry

end midpoint_property_l1146_114604


namespace root_expression_value_l1146_114606

theorem root_expression_value (a : ℝ) : 
  2 * a^2 - 7 * a - 1 = 0 → a * (2 * a - 7) + 5 = 6 := by
  sorry

end root_expression_value_l1146_114606


namespace christina_bank_transfer_l1146_114683

/-- Calculates the remaining balance after a transfer --/
def remaining_balance (initial : ℕ) (transfer : ℕ) : ℕ :=
  initial - transfer

theorem christina_bank_transfer :
  remaining_balance 27004 69 = 26935 := by
  sorry

end christina_bank_transfer_l1146_114683


namespace cherry_pie_count_l1146_114617

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 4 →
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end cherry_pie_count_l1146_114617


namespace fraction_decomposition_l1146_114607

theorem fraction_decomposition (x C D : ℚ) : 
  (7 * x - 15) / (3 * x^2 - x - 4) = C / (x - 1) + D / (3 * x + 4) →
  3 * x^2 - x - 4 = (3 * x + 4) * (x - 1) →
  C = -8/7 ∧ D = 73/7 := by
sorry

end fraction_decomposition_l1146_114607


namespace work_completion_time_l1146_114635

theorem work_completion_time (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 20)
  (hy : y_days = 16)
  (hw : y_worked_days = 12) : 
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 5 := by
  sorry

end work_completion_time_l1146_114635


namespace series_one_over_sqrt_n_diverges_l1146_114691

theorem series_one_over_sqrt_n_diverges :
  ¬ Summable (fun n : ℕ => 1 / Real.sqrt n) := by sorry

end series_one_over_sqrt_n_diverges_l1146_114691


namespace contradiction_assumption_l1146_114652

theorem contradiction_assumption (x y : ℝ) (h : x + y > 2) :
  ¬(x ≤ 1 ∧ y ≤ 1) → (x > 1 ∨ y > 1) := by
  sorry

#check contradiction_assumption

end contradiction_assumption_l1146_114652


namespace hexagon_area_for_given_triangle_l1146_114665

/-- Given an isosceles triangle PQR with circumcircle radius r and perimeter p,
    calculate the area of the hexagon formed by the intersections of the
    perpendicular bisectors of the sides with the circumcircle. -/
def hexagon_area (r p : ℝ) : ℝ :=
  5 * p

theorem hexagon_area_for_given_triangle :
  hexagon_area 10 42 = 210 := by sorry

end hexagon_area_for_given_triangle_l1146_114665


namespace wallet_cost_l1146_114614

theorem wallet_cost (W : ℝ) : 
  W / 2 + 15 + 2 * 15 + 5 = W → W = 100 := by
  sorry

end wallet_cost_l1146_114614


namespace subset_sum_theorem_l1146_114615

theorem subset_sum_theorem (A : Finset ℤ) (h_card : A.card = 4) 
  (h_order : ∃ (a₁ a₂ a₃ a₄ : ℤ), A = {a₁, a₂, a₃, a₄} ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄) 
  (h_subset_sums : (A.powerset.filter (fun s => s.card = 3)).image (fun s => s.sum id) = {-1, 3, 5, 8}) :
  A = {-3, 0, 2, 6} := by
sorry

end subset_sum_theorem_l1146_114615


namespace range_of_m_l1146_114686

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_sol : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) ↔ (m < -1 ∨ m > 4) := by sorry

end range_of_m_l1146_114686


namespace odd_function_decomposition_l1146_114698

/-- An odd function. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A periodic function with period T. -/
def PeriodicFunction (φ : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, φ (x + T) = φ x

/-- A linear function. -/
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∃ k h : ℝ, ∀ x, g x = k * x + h

/-- A function with a center of symmetry at (a, b). -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

theorem odd_function_decomposition (f : ℝ → ℝ) :
  OddFunction f →
  (∃ φ g : ℝ → ℝ, ∃ T : ℝ, T ≠ 0 ∧
    PeriodicFunction φ T ∧
    LinearFunction g ∧
    (∀ x, f x = φ x + g x)) ↔
  (∃ a b : ℝ, (a, b) ≠ (0, 0) ∧ HasCenterOfSymmetry f a b ∧ ∃ k : ℝ, b = k * a) :=
sorry

end odd_function_decomposition_l1146_114698


namespace min_value_exponential_function_l1146_114627

theorem min_value_exponential_function :
  ∀ x : ℝ, 4 * Real.exp x + Real.exp (-x) ≥ 4 ∧
  ∃ x₀ : ℝ, 4 * Real.exp x₀ + Real.exp (-x₀) = 4 := by
  sorry

end min_value_exponential_function_l1146_114627


namespace circle_area_theorem_l1146_114672

theorem circle_area_theorem (r : ℝ) (h : 8 / (2 * Real.pi * r) = (2 * r)^2) :
  π * r^2 = Real.pi^(1/3) := by
  sorry

end circle_area_theorem_l1146_114672


namespace min_value_expression_min_value_achievable_l1146_114624

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) = 216 := by
  sorry

end min_value_expression_min_value_achievable_l1146_114624


namespace math_club_teams_count_l1146_114650

-- Define the number of girls and boys in the math club
def num_girls : ℕ := 5
def num_boys : ℕ := 7

-- Define the number of girls and boys required for each team
def girls_per_team : ℕ := 2
def boys_per_team : ℕ := 2

-- Define the theorem
theorem math_club_teams_count :
  (Nat.choose num_girls girls_per_team) *
  (Nat.choose num_boys boys_per_team) *
  boys_per_team = 420 := by
sorry

end math_club_teams_count_l1146_114650


namespace bubble_bath_per_guest_l1146_114677

theorem bubble_bath_per_guest (couple_rooms : ℕ) (single_rooms : ℕ) (total_bubble_bath : ℕ) :
  couple_rooms = 13 →
  single_rooms = 14 →
  total_bubble_bath = 400 →
  (total_bubble_bath : ℚ) / (2 * couple_rooms + single_rooms) = 10 :=
by sorry

end bubble_bath_per_guest_l1146_114677


namespace pullups_calculation_l1146_114685

/-- Calculates the number of pull-ups done per visit given the total pull-ups per week and visits per day -/
def pullups_per_visit (total_pullups : ℕ) (visits_per_day : ℕ) : ℚ :=
  total_pullups / (visits_per_day * 7)

/-- Theorem: If a person does 70 pull-ups per week and visits a room 5 times per day, 
    then the number of pull-ups done each visit is 2 -/
theorem pullups_calculation :
  pullups_per_visit 70 5 = 2 := by
  sorry

end pullups_calculation_l1146_114685


namespace distance_minimization_l1146_114699

theorem distance_minimization (t : ℝ) (h : t > 0) :
  let f (x : ℝ) := x^2 + 1
  let g (x : ℝ) := Real.log x
  let distance_squared (x : ℝ) := (f x - g x)^2
  (∀ x > 0, distance_squared t ≤ distance_squared x) →
  t = Real.sqrt 2 / 2 :=
by sorry

end distance_minimization_l1146_114699


namespace complex_equation_solution_l1146_114631

theorem complex_equation_solution (z : ℂ) : z * (1 - 2*I) = 2 + I → z = I := by
  sorry

end complex_equation_solution_l1146_114631


namespace least_addition_for_divisibility_l1146_114632

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ (1057 + x) % 23 = 0 ∧ ∀ y : ℕ, y > 0 ∧ (1057 + y) % 23 = 0 → x ≤ y :=
by sorry

end least_addition_for_divisibility_l1146_114632


namespace intersection_of_M_and_N_l1146_114660

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 1} := by sorry

end intersection_of_M_and_N_l1146_114660


namespace square_difference_equals_690_l1146_114609

theorem square_difference_equals_690 : (23 + 15)^2 - (23^2 + 15^2) = 690 := by
  sorry

end square_difference_equals_690_l1146_114609


namespace arithmetic_seq_fifth_term_l1146_114689

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_fifth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_sixth : a 6 = 8) : 
  a 5 = 14 := by
sorry

end arithmetic_seq_fifth_term_l1146_114689


namespace arccos_sqrt3_over_2_l1146_114676

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end arccos_sqrt3_over_2_l1146_114676


namespace smallest_multiple_l1146_114629

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) ∧ 
  (∀ (b : ℕ), b > 0 ∧ 
    (b % 5 = 0) ∧ 
    ((b + 1) % 7 = 0) ∧ 
    ((b + 2) % 9 = 0) ∧ 
    ((b + 3) % 11 = 0) → 
    a ≤ b) ∧
  a = 720 :=
by sorry

end smallest_multiple_l1146_114629


namespace reflect_F_theorem_l1146_114693

/-- Reflects a point over the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The composition of two reflections -/
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (reflect_x_axis p)

theorem reflect_F_theorem :
  let F : ℝ × ℝ := (1, 1)
  double_reflect F = (-1, 1) := by
  sorry

end reflect_F_theorem_l1146_114693


namespace percentage_relation_l1146_114622

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100))
  (hj : j > 0) (hk : k > 0) (hl : l > 0) (hm : m > 0) :
  x = 500 := by
sorry

end percentage_relation_l1146_114622


namespace inequality_solution_range_l1146_114651

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + 2 * (a - 1) * x - 4 < 0) ↔ 
  (-3 < a ∧ a ≤ 1) :=
by sorry

end inequality_solution_range_l1146_114651


namespace min_value_problem_l1146_114670

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (9 * z) / (3 * x + y) + (9 * x) / (y + 3 * z) + (4 * y) / (x + z) ≥ 3 := by
  sorry

end min_value_problem_l1146_114670


namespace coefficient_x_cubed_in_expansion_l1146_114653

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 21).sum (fun k => (Nat.choose 20 k) * (2^(20 - k)) * (if k = 3 then 1 else 0)) = 149462016 := by
  sorry

end coefficient_x_cubed_in_expansion_l1146_114653


namespace base_twelve_square_l1146_114636

theorem base_twelve_square (b : ℕ) : b > 0 → (3 * b + 2)^2 = b^3 + 2 * b^2 + 4 → b = 12 := by
  sorry

end base_twelve_square_l1146_114636


namespace arm_wrestling_tournament_rounds_l1146_114645

/-- Represents the rules and structure of the arm wrestling tournament. -/
structure TournamentRules where
  num_athletes : ℕ
  max_point_diff : ℕ

/-- Calculates the minimum number of rounds required to determine a sole leader. -/
def min_rounds_required (rules : TournamentRules) : ℕ :=
  sorry

/-- Theorem stating that for a tournament with 510 athletes and the given rules,
    the minimum number of rounds required is 9. -/
theorem arm_wrestling_tournament_rounds 
  (rules : TournamentRules) 
  (h1 : rules.num_athletes = 510) 
  (h2 : rules.max_point_diff = 1) : 
  min_rounds_required rules = 9 := by
  sorry

end arm_wrestling_tournament_rounds_l1146_114645


namespace system_solution_l1146_114608

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 2 * y = 8) ∧ 
  (x + 3 * y = 9) ∧ 
  (x = 42 / 11) ∧ 
  (y = 19 / 11) := by
  sorry

end system_solution_l1146_114608


namespace circle_line_distance_l1146_114678

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 + a = 0}
  let center : ℝ × ℝ := (1, 2)  -- Derived from completing the square
  let distance := |1 - 2 + a| / Real.sqrt 2
  (∀ p ∈ circle, p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0) →
  (∀ p ∈ line, p.1 - p.2 + a = 0) →
  distance = Real.sqrt 2 / 2 →
  a = 2 ∨ a = 0 :=
by sorry

end circle_line_distance_l1146_114678


namespace rectangle_area_18_pairs_l1146_114673

theorem rectangle_area_18_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18} = 
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end rectangle_area_18_pairs_l1146_114673


namespace math_team_selection_ways_l1146_114602

def num_boys : ℕ := 6
def num_girls : ℕ := 8
def team_size : ℕ := 4
def min_boys : ℕ := 2

theorem math_team_selection_ways :
  (Finset.sum (Finset.range (team_size - min_boys + 1))
    (fun k => Nat.choose num_boys (min_boys + k) * Nat.choose num_girls (team_size - (min_boys + k)))) = 595 := by
  sorry

end math_team_selection_ways_l1146_114602
