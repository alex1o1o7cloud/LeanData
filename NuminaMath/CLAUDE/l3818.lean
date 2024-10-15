import Mathlib

namespace NUMINAMATH_CALUDE_percentage_equality_l3818_381849

theorem percentage_equality (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3818_381849


namespace NUMINAMATH_CALUDE_number_puzzle_l3818_381809

theorem number_puzzle : ∃ x : ℤ, x - (28 - (37 - (15 - 19))) = 58 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3818_381809


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3818_381813

/-- The distance between the foci of a hyperbola defined by xy = 4 is 8 -/
theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧ 
    ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2)^(1/2 : ℝ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3818_381813


namespace NUMINAMATH_CALUDE_parallel_to_line_if_equal_perpendicular_distances_l3818_381865

structure Geometry2D where
  Point : Type
  Line : Type
  perpendicular_distance : Point → Line → ℝ
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

variable {G : Geometry2D}

theorem parallel_to_line_if_equal_perpendicular_distances
  (A B : G.Point) (l : G.Line) :
  G.perpendicular_distance A l = G.perpendicular_distance B l →
  ∃ (AB : G.Line), G.on_line A AB ∧ G.on_line B AB ∧ G.parallel AB l :=
sorry

end NUMINAMATH_CALUDE_parallel_to_line_if_equal_perpendicular_distances_l3818_381865


namespace NUMINAMATH_CALUDE_pizza_problem_l3818_381812

theorem pizza_problem (slices_per_pizza : ℕ) (games_played : ℕ) (avg_goals_per_game : ℕ) :
  slices_per_pizza = 12 →
  games_played = 8 →
  avg_goals_per_game = 9 →
  (games_played * avg_goals_per_game) / slices_per_pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l3818_381812


namespace NUMINAMATH_CALUDE_megatek_rd_percentage_l3818_381891

theorem megatek_rd_percentage :
  ∀ (manufacturing_angle hr_angle sales_angle rd_angle : ℝ),
  manufacturing_angle = 54 →
  hr_angle = 2 * manufacturing_angle →
  sales_angle = (1/2) * hr_angle →
  rd_angle = 360 - (manufacturing_angle + hr_angle + sales_angle) →
  (rd_angle / 360) * 100 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_megatek_rd_percentage_l3818_381891


namespace NUMINAMATH_CALUDE_smallest_sum_of_consecutive_integers_l3818_381831

theorem smallest_sum_of_consecutive_integers : ∃ n : ℕ,
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 20 * m + 190 = 2 * k^2)) ∧
  (∃ k : ℕ, 20 * n + 190 = 2 * k^2) ∧
  20 * n + 190 = 450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_consecutive_integers_l3818_381831


namespace NUMINAMATH_CALUDE_lowest_price_theorem_l3818_381884

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_theorem (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_costs num_components =
  (production_cost * num_components + shipping_cost * num_components + fixed_costs) / num_components :=
by sorry

#eval lowest_price_per_component 80 2 16200 150

end NUMINAMATH_CALUDE_lowest_price_theorem_l3818_381884


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3818_381852

/-- A regular polygon with perimeter 150 cm and side length 10 cm has 15 sides -/
theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (num_sides : ℕ) :
  perimeter = 150 ∧ side_length = 10 ∧ perimeter = num_sides * side_length → num_sides = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3818_381852


namespace NUMINAMATH_CALUDE_males_in_choir_is_twelve_l3818_381842

/-- Represents the number of musicians in each group -/
structure MusicianCounts where
  orchestra_males : ℕ
  orchestra_females : ℕ
  choir_females : ℕ
  total_musicians : ℕ

/-- Calculates the number of males in the choir based on given conditions -/
def males_in_choir (counts : MusicianCounts) : ℕ :=
  let orchestra_total := counts.orchestra_males + counts.orchestra_females
  let band_total := 2 * orchestra_total
  let choir_total := counts.total_musicians - (orchestra_total + band_total)
  choir_total - counts.choir_females

/-- Theorem stating that the number of males in the choir is 12 -/
theorem males_in_choir_is_twelve (counts : MusicianCounts)
  (h1 : counts.orchestra_males = 11)
  (h2 : counts.orchestra_females = 12)
  (h3 : counts.choir_females = 17)
  (h4 : counts.total_musicians = 98) :
  males_in_choir counts = 12 := by
  sorry

#eval males_in_choir ⟨11, 12, 17, 98⟩

end NUMINAMATH_CALUDE_males_in_choir_is_twelve_l3818_381842


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l3818_381890

/-- Proves that if a fruit seller sells 60% of their apples and has 300 apples left, 
    then they originally had 750 apples. -/
theorem fruit_seller_apples (original : ℕ) (sold_percent : ℚ) (remaining : ℕ) 
    (h1 : sold_percent = 60 / 100)
    (h2 : remaining = 300)
    (h3 : (1 - sold_percent) * original = remaining) : 
  original = 750 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l3818_381890


namespace NUMINAMATH_CALUDE_crane_sling_diameter_l3818_381817

/-- Represents the problem of finding the smallest suitable rope diameter for a crane sling --/
theorem crane_sling_diameter
  (M : ℝ)  -- Mass of the load in tons
  (n : ℕ)  -- Number of slings
  (α : ℝ)  -- Angle of each sling with vertical in radians
  (k : ℝ)  -- Safety factor
  (q : ℝ)  -- Maximum load per thread in N/mm²
  (g : ℝ)  -- Free fall acceleration in m/s²
  (h₁ : M = 20)
  (h₂ : n = 3)
  (h₃ : α = Real.pi / 6)  -- 30° in radians
  (h₄ : k = 6)
  (h₅ : q = 1000)
  (h₆ : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    (∀ (D' : ℕ), D' < D → 
      (Real.pi * D'^2 / 4) * q * 10^6 < 
      k * M * g * 1000 / (n * Real.cos α)) ∧
    (Real.pi * D^2 / 4) * q * 10^6 ≥ 
    k * M * g * 1000 / (n * Real.cos α) :=
sorry

end NUMINAMATH_CALUDE_crane_sling_diameter_l3818_381817


namespace NUMINAMATH_CALUDE_subsets_without_consecutive_eq_fib_l3818_381832

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of subsets without consecutive elements -/
def subsets_without_consecutive (n : ℕ) : ℕ :=
  fib (n + 2)

/-- Theorem: The number of subsets of {1, 2, 3, ..., n} that do not contain
    two consecutive numbers is equal to the (n+2)th Fibonacci number -/
theorem subsets_without_consecutive_eq_fib (n : ℕ) :
  subsets_without_consecutive n = fib (n + 2) := by
  sorry


end NUMINAMATH_CALUDE_subsets_without_consecutive_eq_fib_l3818_381832


namespace NUMINAMATH_CALUDE_phillip_test_results_l3818_381854

/-- Represents the number of questions Phillip gets right on a test -/
def correct_answers (total : ℕ) (percentage : ℚ) : ℚ :=
  (total : ℚ) * percentage

/-- Represents the total number of correct answers across all tests -/
def total_correct_answers (x : ℕ) : ℚ :=
  correct_answers 40 (75/100) + correct_answers 50 (98/100) + (x : ℚ) * ((100 - x : ℚ)/100)

theorem phillip_test_results (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  total_correct_answers x = 79 + (x : ℚ) * ((100 - x : ℚ)/100) :=
by sorry

end NUMINAMATH_CALUDE_phillip_test_results_l3818_381854


namespace NUMINAMATH_CALUDE_candy_count_l3818_381882

/-- The number of candies initially in the pile -/
def initial_candies : ℕ := 6

/-- The number of candies added to the pile -/
def added_candies : ℕ := 4

/-- The total number of candies after adding -/
def total_candies : ℕ := initial_candies + added_candies

theorem candy_count : total_candies = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3818_381882


namespace NUMINAMATH_CALUDE_concert_attendance_l3818_381894

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l3818_381894


namespace NUMINAMATH_CALUDE_residue_of_7_2050_mod_19_l3818_381839

theorem residue_of_7_2050_mod_19 : 7^2050 % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_2050_mod_19_l3818_381839


namespace NUMINAMATH_CALUDE_f₁_solution_set_f₂_min_value_l3818_381886

-- Part 1
def f₁ (x : ℝ) : ℝ := |3*x - 1| + |x + 3|

theorem f₁_solution_set : 
  {x : ℝ | f₁ x ≥ 4} = {x : ℝ | x ≤ -3 ∨ x ≥ 1/2} := by sorry

-- Part 2
def f₂ (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem f₂_min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (hmin : ∃ x, ∀ y, f₂ b c x ≤ f₂ b c y) 
  (hval : ∃ x, f₂ b c x = 1) : 
  (1/b + 1/c) ≥ 4 ∧ ∃ b c, (1/b + 1/c = 4 ∧ b > 0 ∧ c > 0) := by sorry

end NUMINAMATH_CALUDE_f₁_solution_set_f₂_min_value_l3818_381886


namespace NUMINAMATH_CALUDE_smallest_number_of_ducks_l3818_381810

def duck_flock_size : ℕ := 18
def seagull_flock_size : ℕ := 10

theorem smallest_number_of_ducks (total_ducks total_seagulls : ℕ) : 
  total_ducks = total_seagulls → 
  total_ducks % duck_flock_size = 0 →
  total_seagulls % seagull_flock_size = 0 →
  total_ducks ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_ducks_l3818_381810


namespace NUMINAMATH_CALUDE_exam_probability_theorem_l3818_381868

def exam_probability (P : ℝ) : Prop :=
  let correct_prob : List ℝ := [1, P, 1/2, 1/4]
  let perfect_score_prob := List.prod correct_prob
  let prob_15 := P * (1/2) * (3/4) + (1-P) * (1/2) * (3/4)
  let prob_10 := P * (1/2) * (3/4) * (1/4) + (1-P) * (1/2) * (1/4) + (1-P) * (1/2) * (3/4) * (1/4)
  let prob_5 := P * (1/2) * (3/4) + (1-P) * (1/2) * (1/4)
  let prob_0 := 1 - perfect_score_prob - prob_15 - prob_10 - prob_5
  (P = 2/3 → perfect_score_prob = 1/12) ∧
  (prob_15 = 1/8 ∧ prob_10 = 1/8) ∧
  (prob_0 = 1/6)

theorem exam_probability_theorem : exam_probability (2/3) :=
sorry

end NUMINAMATH_CALUDE_exam_probability_theorem_l3818_381868


namespace NUMINAMATH_CALUDE_g_evaluation_l3818_381856

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l3818_381856


namespace NUMINAMATH_CALUDE_f_lower_bound_g_inequality_min_a_l3818_381818

noncomputable section

variables (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := a * Real.exp (2*x - 1) - x^2 * (Real.log x + 1/2)

def g (x : ℝ) (a : ℝ) : ℝ := x * f x a + x^2 / Real.exp x

theorem f_lower_bound (h : x > 0) : f x 0 ≥ x^2/2 - x^3 := by sorry

theorem g_inequality_min_a :
  (∀ x > 1, x * g (Real.log x / (x - 1)) a < g (x * Real.log x / (x - 1)) a) ↔ a ≥ 1 / Real.exp 1 := by sorry

end

end NUMINAMATH_CALUDE_f_lower_bound_g_inequality_min_a_l3818_381818


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3818_381808

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k : ℕ, k > 0 ∧ k < n → ¬(6^k ≡ k^6 [ZMOD 3])) ∧ 
  (6^n ≡ n^6 [ZMOD 3]) → 
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3818_381808


namespace NUMINAMATH_CALUDE_log_stack_sum_l3818_381857

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 4 logs
This theorem proves that the total number of logs in the stack is 114.
-/
theorem log_stack_sum : ∀ (n : ℕ) (a l : ℤ),
  n = 15 - 4 + 1 →
  a = 15 →
  l = 4 →
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3818_381857


namespace NUMINAMATH_CALUDE_string_length_problem_l3818_381827

theorem string_length_problem (num_strings : ℕ) (total_length : ℝ) (h1 : num_strings = 7) (h2 : total_length = 98) :
  total_length / num_strings = 14 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l3818_381827


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3818_381895

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_per_mile : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

#check driver_net_rate_of_pay

end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3818_381895


namespace NUMINAMATH_CALUDE_fernandez_family_has_nine_children_l3818_381861

/-- Represents the Fernandez family structure and ages -/
structure FernandezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : ℕ → ℕ
  average_family_age : ℕ
  father_age : ℕ
  grandmother_age : ℕ
  average_mother_children_age : ℕ

/-- The Fernandez family satisfies the given conditions -/
def is_valid_fernandez_family (f : FernandezFamily) : Prop :=
  f.average_family_age = 25 ∧
  f.father_age = 50 ∧
  f.grandmother_age = 70 ∧
  f.average_mother_children_age = 18

/-- The theorem stating that the Fernandez family has 9 children -/
theorem fernandez_family_has_nine_children (f : FernandezFamily) 
  (h : is_valid_fernandez_family f) : f.num_children = 9 := by
  sorry

#check fernandez_family_has_nine_children

end NUMINAMATH_CALUDE_fernandez_family_has_nine_children_l3818_381861


namespace NUMINAMATH_CALUDE_annual_compound_interest_rate_exists_l3818_381883

-- Define the initial principal
def initial_principal : ℝ := 780

-- Define the final amount
def final_amount : ℝ := 1300

-- Define the time period in years
def time_period : ℕ := 4

-- Define the compound interest equation
def compound_interest_equation (r : ℝ) : Prop :=
  final_amount = initial_principal * (1 + r) ^ time_period

-- Theorem statement
theorem annual_compound_interest_rate_exists :
  ∃ r : ℝ, compound_interest_equation r ∧ r > 0 ∧ r < 1 :=
sorry

end NUMINAMATH_CALUDE_annual_compound_interest_rate_exists_l3818_381883


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3818_381834

theorem complex_equation_solution (z : ℂ) : 
  z^2 + 2*Complex.I*z + 3 = 0 ↔ z = Complex.I ∨ z = -3*Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3818_381834


namespace NUMINAMATH_CALUDE_garden_constant_value_l3818_381863

/-- Represents a square garden with area and perimeter --/
structure SquareGarden where
  area : ℝ
  perimeter : ℝ

/-- The constant in the relationship between area and perimeter --/
def garden_constant (g : SquareGarden) : ℝ :=
  g.area - 2 * g.perimeter

theorem garden_constant_value :
  ∀ g : SquareGarden,
    g.area = g.perimeter^2 / 16 →
    g.area = 2 * g.perimeter + garden_constant g →
    g.perimeter = 38 →
    garden_constant g = 14.25 := by
  sorry

end NUMINAMATH_CALUDE_garden_constant_value_l3818_381863


namespace NUMINAMATH_CALUDE_complement_of_40_30_l3818_381877

/-- The complement of an angle is the difference between 90 degrees and the angle. -/
def complementOfAngle (angle : ℚ) : ℚ := 90 - angle

/-- Represents 40 degrees and 30 minutes in decimal degrees. -/
def angleA : ℚ := 40 + 30 / 60

theorem complement_of_40_30 :
  complementOfAngle angleA = 49 + 30 / 60 := by sorry

end NUMINAMATH_CALUDE_complement_of_40_30_l3818_381877


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l3818_381824

theorem mashed_potatoes_count :
  let bacon_count : ℕ := 42
  let difference : ℕ := 366
  let mashed_potatoes_count : ℕ := bacon_count + difference
  mashed_potatoes_count = 408 := by
sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l3818_381824


namespace NUMINAMATH_CALUDE_f_two_thirds_eq_three_halves_l3818_381870

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (y : ℝ) : ℝ := y / ((2 - y) / 3)

theorem f_two_thirds_eq_three_halves :
  f (2/3) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_f_two_thirds_eq_three_halves_l3818_381870


namespace NUMINAMATH_CALUDE_flash_interval_value_l3818_381823

/-- The number of flashes in ¾ of an hour -/
def flashes : ℕ := 240

/-- The duration in hours -/
def duration : ℚ := 3/4

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time interval between flashes in seconds -/
def flash_interval : ℚ := (duration * seconds_per_hour) / flashes

theorem flash_interval_value : flash_interval = 45/4 := by sorry

end NUMINAMATH_CALUDE_flash_interval_value_l3818_381823


namespace NUMINAMATH_CALUDE_sallys_peaches_l3818_381869

/-- The total number of peaches Sally has after picking more from the orchard -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that given Sally's initial 13 peaches and her additional 55 picked peaches, 
    the total number of peaches is 68 -/
theorem sallys_peaches : total_peaches 13 55 = 68 := by
  sorry

end NUMINAMATH_CALUDE_sallys_peaches_l3818_381869


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l3818_381859

theorem three_numbers_ratio (a b c : ℕ+) : 
  (Nat.lcm a (Nat.lcm b c) = 2400) → 
  (Nat.gcd a (Nat.gcd b c) = 40) → 
  (∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k) := by
sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l3818_381859


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3818_381898

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The quadrilateral has a circumscribed circle -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Bool := sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral,
    q.perpendicular_diagonals ∧
    q.has_inscribed_circle ∧
    q.has_circumscribed_circle ∧
    ¬(is_square q) :=
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3818_381898


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3818_381822

/-- The lateral surface area of a cone with base radius 6 and volume 30π is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3818_381822


namespace NUMINAMATH_CALUDE_problem_statement_l3818_381825

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1)^2 = 0 → 3*b - a = -5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3818_381825


namespace NUMINAMATH_CALUDE_tom_four_times_cindy_l3818_381873

/-- Tom's current age -/
def t : ℕ := sorry

/-- Cindy's current age -/
def c : ℕ := sorry

/-- In five years, Tom will be twice as old as Cindy -/
axiom future_condition : t + 5 = 2 * (c + 5)

/-- Thirteen years ago, Tom was three times as old as Cindy -/
axiom past_condition : t - 13 = 3 * (c - 13)

/-- The number of years ago when Tom was four times as old as Cindy -/
def years_ago : ℕ := sorry

theorem tom_four_times_cindy : years_ago = 19 := by sorry

end NUMINAMATH_CALUDE_tom_four_times_cindy_l3818_381873


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_zero_solution_l3818_381892

theorem smallest_n_for_unique_zero_solution :
  ∃ (n : ℕ), n ≥ 1 ∧
  (∀ (a b c d : ℤ), a^2 + b^2 + c^2 - n * d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (a b c d : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) ∧ a^2 + b^2 + c^2 - m * d^2 = 0) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_zero_solution_l3818_381892


namespace NUMINAMATH_CALUDE_container_max_volume_l3818_381850

/-- The volume function for the container --/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function --/
def volume_derivative (x : ℝ) : ℝ := 12 * (x^2 - 46*x + 360)

theorem container_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  ∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x ∧
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_container_max_volume_l3818_381850


namespace NUMINAMATH_CALUDE_kanul_cash_proof_l3818_381811

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 1000

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 500

/-- The amount spent on machinery -/
def machinery : ℝ := 400

/-- The percentage of total cash spent as cash -/
def cash_percentage : ℝ := 0.1

theorem kanul_cash_proof :
  total_cash = raw_materials + machinery + cash_percentage * total_cash :=
by sorry

end NUMINAMATH_CALUDE_kanul_cash_proof_l3818_381811


namespace NUMINAMATH_CALUDE_smallest_n_with_square_sums_l3818_381885

theorem smallest_n_with_square_sums : ∃ (a b c : ℕ), 
  (a < b ∧ b < c) ∧ 
  (∃ (x y z : ℕ), a + b = x^2 ∧ a + c = y^2 ∧ b + c = z^2) ∧
  a + b + c = 55 ∧
  (∀ (n : ℕ), n < 55 → 
    ¬∃ (a' b' c' : ℕ), (a' < b' ∧ b' < c') ∧ 
    (∃ (x' y' z' : ℕ), a' + b' = x'^2 ∧ a' + c' = y'^2 ∧ b' + c' = z'^2) ∧
    a' + b' + c' = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_square_sums_l3818_381885


namespace NUMINAMATH_CALUDE_odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l3818_381837

-- Define the function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_composition (h : ∀ x, f (-x) = -f x) : ∀ x, (f ∘ f) (-x) = -(f ∘ f) x :=
sorry

-- Statement 2
theorem periodic_function_composition (h : ∃ T, ∀ x, f (x + T) = f x) : 
  ∃ T, ∀ x, (f ∘ f) (x + T) = (f ∘ f) x :=
sorry

-- Statement 3
theorem exists_non_decreasing_composition :
  ∃ f : ℝ → ℝ, (∀ x y, x < y → f x > f y) ∧ ¬(∀ x y, x < y → (f ∘ f) x > (f ∘ f) y) :=
sorry

-- Statement 4
theorem inverse_function_zero (h₁ : Function.Bijective f) 
  (h₂ : ∃ x, f x = Function.invFun f x) : ∃ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l3818_381837


namespace NUMINAMATH_CALUDE_saltwater_concentration_l3818_381897

/-- Represents a saltwater solution -/
structure SaltWaterSolution where
  salt : ℝ
  water : ℝ
  concentration : ℝ
  concentration_def : concentration = salt / (salt + water) * 100

/-- The condition that adding 200g of water halves the concentration -/
def half_concentration (s : SaltWaterSolution) : Prop :=
  s.salt / (s.salt + s.water + 200) = s.concentration / 2

/-- The condition that adding 25g of salt doubles the concentration -/
def double_concentration (s : SaltWaterSolution) : Prop :=
  (s.salt + 25) / (s.salt + s.water + 25) = 2 * s.concentration / 100

/-- The main theorem to prove -/
theorem saltwater_concentration 
  (s : SaltWaterSolution) 
  (h1 : half_concentration s) 
  (h2 : double_concentration s) : 
  s.concentration = 10 := by
  sorry


end NUMINAMATH_CALUDE_saltwater_concentration_l3818_381897


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l3818_381872

theorem polynomial_perfect_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l3818_381872


namespace NUMINAMATH_CALUDE_water_collection_impossible_l3818_381875

def total_water (n : ℕ) : ℕ := n * (n + 1) / 2

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem water_collection_impossible (n : ℕ) (h : n = 2018) :
  is_odd (total_water n) ∧ 
  (∀ m : ℕ, m ≤ n → ∃ k, m = 2 ^ k) → False :=
by sorry

end NUMINAMATH_CALUDE_water_collection_impossible_l3818_381875


namespace NUMINAMATH_CALUDE_equation_solution_implies_a_range_l3818_381844

theorem equation_solution_implies_a_range (a : ℝ) :
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2 * a - 1) →
  -1/2 ≤ a ∧ a ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_implies_a_range_l3818_381844


namespace NUMINAMATH_CALUDE_cd_ratio_l3818_381801

/-- Represents the number of CDs Tyler has at different stages --/
structure CDCount where
  initial : ℕ
  given_away : ℕ
  bought : ℕ
  final : ℕ

/-- Theorem stating the ratio of CDs given away to initial CDs --/
theorem cd_ratio (c : CDCount) 
  (h1 : c.initial = 21)
  (h2 : c.bought = 8)
  (h3 : c.final = 22)
  (h4 : c.initial - c.given_away + c.bought = c.final) :
  (c.given_away : ℚ) / c.initial = 1 / 3 := by
  sorry

#check cd_ratio

end NUMINAMATH_CALUDE_cd_ratio_l3818_381801


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3818_381893

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 4) ↔ 
  (-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ 
  (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2) := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3818_381893


namespace NUMINAMATH_CALUDE_equation_equivalence_l3818_381851

theorem equation_equivalence (x Q : ℝ) (h : 5 * (5 * x + 7 * Real.pi) = Q) :
  10 * (10 * x + 14 * Real.pi + 2) = 4 * Q + 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3818_381851


namespace NUMINAMATH_CALUDE_parabola_point_relation_l3818_381887

-- Define the parabola function
def parabola (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- Define the theorem
theorem parabola_point_relation (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : parabola c (-4) = y₁)
  (h2 : parabola c (-2) = y₂)
  (h3 : parabola c (1/2) = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l3818_381887


namespace NUMINAMATH_CALUDE_five_people_handshakes_l3818_381874

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 5 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 10. -/
theorem five_people_handshakes : number_of_handshakes 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_people_handshakes_l3818_381874


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3818_381878

def digit_repeat (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^100 - 1) / 9

def two_digit_repeat (d₁ d₂ : ℕ) (n : ℕ) : ℕ :=
  (10 * d₁ + d₂) * (10^99 - 1) / 99 + d₁ * 10^99

def N₁ : ℕ := digit_repeat 3 100
def N₂ : ℕ := digit_repeat 6 100
def N₃ : ℕ := two_digit_repeat 1 5 100
def N₄ : ℕ := two_digit_repeat 2 1 100
def N₅ : ℕ := two_digit_repeat 2 7 100

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem no_perfect_squares :
  ¬(is_perfect_square N₁ ∨ is_perfect_square N₂ ∨ is_perfect_square N₃ ∨ is_perfect_square N₄ ∨ is_perfect_square N₅) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3818_381878


namespace NUMINAMATH_CALUDE_intersection_point_equality_l3818_381862

theorem intersection_point_equality (a b c d : ℝ) : 
  (1 = 1^2 + a * 1 + b) → 
  (1 = 1^2 + c * 1 + d) → 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equality_l3818_381862


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l3818_381896

theorem different_tens_digit_probability : 
  let total_integers : ℕ := 70
  let chosen_integers : ℕ := 7
  let tens_digits : ℕ := 7
  let integers_per_tens : ℕ := 10

  let favorable_outcomes : ℕ := integers_per_tens ^ chosen_integers
  let total_outcomes : ℕ := Nat.choose total_integers chosen_integers

  (favorable_outcomes : ℚ) / total_outcomes = 20000 / 83342961 := by
  sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l3818_381896


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3818_381821

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + i) = 1 - i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3818_381821


namespace NUMINAMATH_CALUDE_pizza_distribution_l3818_381814

/-- Given 6 people sharing 3 pizzas with 8 slices each, if they all eat the same amount and finish all the pizzas, each person will eat 4 slices. -/
theorem pizza_distribution (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 :=
by
  sorry

#check pizza_distribution

end NUMINAMATH_CALUDE_pizza_distribution_l3818_381814


namespace NUMINAMATH_CALUDE_waiter_tables_count_l3818_381820

/-- Calculates the number of tables a waiter has based on customer information -/
def waiterTables (initialCustomers leavingCustomers peoplePerTable : ℕ) : ℕ :=
  (initialCustomers - leavingCustomers) / peoplePerTable

/-- Theorem stating that under the given conditions, the waiter had 5 tables -/
theorem waiter_tables_count :
  waiterTables 62 17 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_count_l3818_381820


namespace NUMINAMATH_CALUDE_coprime_implies_divisible_power_minus_one_l3818_381888

theorem coprime_implies_divisible_power_minus_one (a n : ℕ) (h : Nat.Coprime a n) :
  ∃ m : ℕ, n ∣ (a^m - 1) := by
sorry

end NUMINAMATH_CALUDE_coprime_implies_divisible_power_minus_one_l3818_381888


namespace NUMINAMATH_CALUDE_remainder_mod_six_l3818_381841

theorem remainder_mod_six (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_six_l3818_381841


namespace NUMINAMATH_CALUDE_problem_solution_l3818_381847

theorem problem_solution (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3818_381847


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_range_l3818_381807

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- Part I
theorem solve_inequality :
  {x : ℝ | f x < 4 - |x - 1|} = {x : ℝ | -5/4 < x ∧ x < 1/2} :=
sorry

-- Part II
theorem find_a_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ (0 < a ∧ a ≤ 10/3) :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_range_l3818_381807


namespace NUMINAMATH_CALUDE_book_discount_l3818_381879

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- The price reduction percentage -/
def reduction_percentage : Rat := 62.5

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat := 10 * n.tens + n.ones

/-- Checks if two TwoDigitNumbers have the same digits in different order -/
def same_digits (n1 n2 : TwoDigitNumber) : Prop :=
  (n1.tens = n2.ones) ∧ (n1.ones = n2.tens)

theorem book_discount (original reduced : TwoDigitNumber)
  (h_reduction : value reduced = (100 - reduction_percentage) / 100 * value original)
  (h_same_digits : same_digits original reduced) :
  value original - value reduced = 45 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_l3818_381879


namespace NUMINAMATH_CALUDE_cos_178_plus_theta_l3818_381803

theorem cos_178_plus_theta (θ : ℝ) (h : Real.sin (88 * π / 180 + θ) = 2/3) :
  Real.cos (178 * π / 180 + θ) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_178_plus_theta_l3818_381803


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l3818_381876

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * I) = 4/5 + 3/5 * I →
  Complex.exp (δ * I) = -5/13 - 12/13 * I →
  Real.cos (γ + δ) = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l3818_381876


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3818_381830

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (∀ y, x = 0 → x * y = 0) ∧ (∃ x y, x * y = 0 ∧ x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3818_381830


namespace NUMINAMATH_CALUDE_pastry_distribution_l3818_381864

/-- The number of pastries the Hatter initially had -/
def total_pastries : ℕ := 32

/-- The fraction of pastries March Hare ate -/
def march_hare_fraction : ℚ := 5/16

/-- The fraction of remaining pastries Dormouse ate -/
def dormouse_fraction : ℚ := 7/11

/-- The number of pastries left for the Hatter -/
def hatter_leftover : ℕ := 8

/-- The number of pastries March Hare ate -/
def march_hare_eaten : ℕ := 10

/-- The number of pastries Dormouse ate -/
def dormouse_eaten : ℕ := 14

theorem pastry_distribution :
  (march_hare_eaten = (total_pastries : ℚ) * march_hare_fraction) ∧
  (dormouse_eaten = ((total_pastries - march_hare_eaten) : ℚ) * dormouse_fraction) ∧
  (hatter_leftover = total_pastries - march_hare_eaten - dormouse_eaten) :=
by sorry

end NUMINAMATH_CALUDE_pastry_distribution_l3818_381864


namespace NUMINAMATH_CALUDE_ellipse_condition_l3818_381829

def is_ellipse_with_y_axis_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m - 1) = 1 ↔ 
    x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  is_ellipse_with_y_axis_foci m ↔ 3 < m ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3818_381829


namespace NUMINAMATH_CALUDE_box_side_length_l3818_381860

/-- The length of one side of a cubic box given total volume, cost per box, and total cost -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (h1 : total_volume = 2.16e6)  -- 2.16 million cubic inches
  (h2 : cost_per_box = 0.5)     -- $0.50 per box
  (h3 : total_cost = 225)       -- $225 total cost
  : ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_box_side_length_l3818_381860


namespace NUMINAMATH_CALUDE_item_sale_ratio_l3818_381858

theorem item_sale_ratio (x y c : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l3818_381858


namespace NUMINAMATH_CALUDE_total_nuts_is_half_cup_l3818_381805

/-- The amount of walnuts Karen added to the trail mix in cups -/
def walnuts : ℚ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def almonds : ℚ := 0.25

/-- The total amount of nuts Karen added to the trail mix in cups -/
def total_nuts : ℚ := walnuts + almonds

/-- Theorem stating that the total amount of nuts Karen added is 0.50 cups -/
theorem total_nuts_is_half_cup : total_nuts = 0.50 := by sorry

end NUMINAMATH_CALUDE_total_nuts_is_half_cup_l3818_381805


namespace NUMINAMATH_CALUDE_room_width_l3818_381816

/-- Given a rectangular room with length 18 m and unknown width, surrounded by a 2 m wide veranda on all sides, 
    if the area of the veranda is 136 m², then the width of the room is 12 m. -/
theorem room_width (w : ℝ) : 
  w > 0 →  -- Ensure width is positive
  (22 * (w + 4) - 18 * w = 136) →  -- Area of veranda equation
  w = 12 := by
sorry

end NUMINAMATH_CALUDE_room_width_l3818_381816


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3818_381855

/-- Given three collinear points in 3D space, prove that the sum of x and y coordinates of two points is -1/2 --/
theorem collinear_points_sum (x y : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 2, 0)
  let B : ℝ × ℝ × ℝ := (x, 3, -1)
  let C : ℝ × ℝ × ℝ := (4, y, 2)
  (∃ (t : ℝ), B - A = t • (C - A)) → x + y = -1/2 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_sum_l3818_381855


namespace NUMINAMATH_CALUDE_club_officer_selection_l3818_381828

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  let boy_president := boys * (boys - 1) * girls
  let girl_president := girls * (girls - 1) * boys
  boy_president + girl_president

/-- The main theorem stating the number of ways to choose officers --/
theorem club_officer_selection :
  choose_officers 24 14 10 = 3080 :=
by sorry

end NUMINAMATH_CALUDE_club_officer_selection_l3818_381828


namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l3818_381806

theorem unique_perfect_square_solution : 
  ∃! (n : ℕ), n > 0 ∧ ∃ (m : ℕ), n^4 - n^3 + 3*n^2 + 5 = m^2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l3818_381806


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l3818_381848

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  -- The actual calculation would go here
  24

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.originalSideLength = 4 ∧ 
    c.removedCubeSideLength = 1 → 
    edgeCount c = 24 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l3818_381848


namespace NUMINAMATH_CALUDE_dave_tickets_l3818_381871

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 18 tickets -/
theorem dave_tickets : final_tickets 25 22 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l3818_381871


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3818_381889

theorem max_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : 
  x^2 + y^2 ≤ 24421 ∧ ∃ (a b : ℤ), a^2 - b^2 = 221 ∧ a^2 + b^2 = 24421 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3818_381889


namespace NUMINAMATH_CALUDE_cost_of_horse_l3818_381835

/-- Proves that the cost of a horse is 2000 given the problem conditions --/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧ 
  horse_profit_rate = 0.1 ∧ 
  cow_profit_rate = 0.2 ∧ 
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_horse_l3818_381835


namespace NUMINAMATH_CALUDE_probability_integer_exponent_x_l3818_381840

theorem probability_integer_exponent_x (a : ℝ) (x : ℝ) :
  let expansion := (x - a / Real.sqrt x) ^ 5
  let total_terms := 6
  let integer_exponent_terms := 3
  (integer_exponent_terms : ℚ) / total_terms = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_probability_integer_exponent_x_l3818_381840


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_in_circle_l3818_381819

/-- Given an isosceles right triangle inscribed in a circle with radius √2,
    where the side lengths are in the ratio 1:1:√2,
    prove that the area of the triangle is 2 and the circumference of the circle is 2π√2. -/
theorem isosceles_right_triangle_in_circle 
  (r : ℝ) 
  (h_r : r = Real.sqrt 2) 
  (a b c : ℝ) 
  (h_abc : a = b ∧ c = a * Real.sqrt 2) 
  (h_inscribed : c = 2 * r) : 
  (1/2 * a * b = 2) ∧ (2 * Real.pi * r = 2 * Real.pi * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_in_circle

end NUMINAMATH_CALUDE_isosceles_right_triangle_in_circle_l3818_381819


namespace NUMINAMATH_CALUDE_number_problem_l3818_381826

theorem number_problem : ∃ x : ℝ, 0.65 * x = 0.05 * 60 + 23 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3818_381826


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l3818_381815

-- Define the percentage as a fraction
def one_third_percent : ℚ := 1 / 3 / 100

-- Define the value we're calculating the percentage of
def base_value : ℚ := 180

-- Theorem statement
theorem one_third_percent_of_180 : one_third_percent * base_value = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l3818_381815


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l3818_381802

theorem absolute_value_of_w (w : ℂ) : w^2 - 6*w + 40 = 0 → Complex.abs w = Real.sqrt 40 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l3818_381802


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3818_381880

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3818_381880


namespace NUMINAMATH_CALUDE_triangle_properties_l3818_381899

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  b * Real.cos A + a / 2 = c ∧
  c = 2 * a ∧
  b = 3 * Real.sqrt 3 →
  B = π / 3 ∧ (a * c * Real.sin B) / 2 = (9 * Real.sqrt 3) / 2 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l3818_381899


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l3818_381853

theorem cucumber_water_percentage (initial_weight initial_water_percentage new_weight : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 99 →
  new_weight = 50 →
  let initial_water := initial_weight * (initial_water_percentage / 100)
  let initial_solid := initial_weight - initial_water
  let new_water := new_weight - initial_solid
  new_water / new_weight * 100 = 98 := by
sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l3818_381853


namespace NUMINAMATH_CALUDE_brad_green_balloons_l3818_381843

/-- Calculates the number of green balloons Brad has -/
def green_balloons (total : ℕ) (initial_red : ℕ) (popped_red : ℕ) (blue : ℕ) : ℕ :=
  let remaining_red := initial_red - popped_red
  let non_red := total - remaining_red
  let green_and_yellow := non_red - blue
  (2 * green_and_yellow) / 5

/-- Theorem stating that Brad has 12 green balloons -/
theorem brad_green_balloons :
  green_balloons 50 15 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brad_green_balloons_l3818_381843


namespace NUMINAMATH_CALUDE_salary_may_value_l3818_381881

def salary_problem (jan feb mar apr may : ℕ) : Prop :=
  (jan + feb + mar + apr) / 4 = 8000 ∧
  (feb + mar + apr + may) / 4 = 8300 ∧
  jan = 5300

theorem salary_may_value :
  ∀ (jan feb mar apr may : ℕ),
    salary_problem jan feb mar apr may →
    may = 6500 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_may_value_l3818_381881


namespace NUMINAMATH_CALUDE_only_cube_has_congruent_views_l3818_381838

-- Define the possible solids
inductive Solid
  | Cone
  | Cylinder
  | Cube
  | SquarePyramid

-- Define a function to check if a solid has congruent views
def hasCongruentViews (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => False
  | Solid.Cube => True
  | Solid.SquarePyramid => False

-- Theorem stating that only a cube has congruent views
theorem only_cube_has_congruent_views :
  ∀ s : Solid, hasCongruentViews s ↔ s = Solid.Cube :=
by sorry

end NUMINAMATH_CALUDE_only_cube_has_congruent_views_l3818_381838


namespace NUMINAMATH_CALUDE_sequence_problem_l3818_381867

theorem sequence_problem (x : ℕ → ℤ) 
  (h1 : x 1 = 8)
  (h2 : x 4 = 2)
  (h3 : ∀ n : ℕ, n > 0 → x (n + 2) + x n = 2 * x (n + 1)) :
  x 10 = -10 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l3818_381867


namespace NUMINAMATH_CALUDE_quadratic_solution_and_gcd_sum_l3818_381836

theorem quadratic_solution_and_gcd_sum : ∃ m n p : ℕ,
  (∀ x : ℝ, x * (4 * x - 5) = 7 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 150 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_gcd_sum_l3818_381836


namespace NUMINAMATH_CALUDE_circle_center_sum_l3818_381845

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 9, 
    the sum of the x and y coordinates of its center is 7 -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 9 → ∃ h k : ℝ, (h, k) = (3, 4) ∧ h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3818_381845


namespace NUMINAMATH_CALUDE_find_k_l3818_381866

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) → k = -18 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3818_381866


namespace NUMINAMATH_CALUDE_stickers_per_page_l3818_381846

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) (h1 : total_stickers = 220) (h2 : total_pages = 22) :
  total_stickers / total_pages = 10 := by
  sorry

end NUMINAMATH_CALUDE_stickers_per_page_l3818_381846


namespace NUMINAMATH_CALUDE_x_value_l3818_381833

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3818_381833


namespace NUMINAMATH_CALUDE_first_train_speed_l3818_381800

/-- Proves that the speed of the first train is 45 kmph given the problem conditions --/
theorem first_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧ v * (1 + t) = 90 ∧ 90 * t = 90) → -- Equations from the problem
  v = 45 := by
  sorry


end NUMINAMATH_CALUDE_first_train_speed_l3818_381800


namespace NUMINAMATH_CALUDE_range_of_m_l3818_381804

theorem range_of_m (m : ℝ) : 
  (¬ (∃ m : ℝ, m + 1 ≤ 0) ∨ ¬ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  (m ≤ -2 ∨ m > -1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3818_381804
