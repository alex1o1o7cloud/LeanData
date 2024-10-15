import Mathlib

namespace NUMINAMATH_CALUDE_triple_integral_equality_l628_62878

open MeasureTheory Interval Set

theorem triple_integral_equality {f : ℝ → ℝ} (hf : ContinuousOn f (Ioo 0 1)) :
  ∫ x in (Icc 0 1), ∫ y in (Icc x 1), ∫ z in (Icc x y), f x * f y * f z = 
  (1 / 6) * (∫ x in (Icc 0 1), f x) ^ 3 := by sorry

end NUMINAMATH_CALUDE_triple_integral_equality_l628_62878


namespace NUMINAMATH_CALUDE_square_formation_for_12_and_15_l628_62844

/-- Given n sticks with lengths 1, 2, ..., n, determine if a square can be formed
    or the minimum number of sticks to be broken in half to form a square. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  let totalLength := n * (n + 1) / 2
  if totalLength % 4 = 0 then 0
  else
    let targetLength := (totalLength / 4 + 1) * 4
    (targetLength - totalLength + 1) / 2

theorem square_formation_for_12_and_15 :
  minSticksToBreak 12 = 2 ∧ minSticksToBreak 15 = 0 := by
  sorry


end NUMINAMATH_CALUDE_square_formation_for_12_and_15_l628_62844


namespace NUMINAMATH_CALUDE_time_to_return_is_45_minutes_l628_62843

/-- Represents a hiker's journey on a trail --/
structure HikerJourney where
  rate : Real  -- Minutes per kilometer
  initialDistance : Real  -- Kilometers hiked east initially
  totalDistance : Real  -- Total kilometers hiked east before turning back
  
/-- Calculates the time required for a hiker to return to the start of the trail --/
def timeToReturn (journey : HikerJourney) : Real :=
  sorry

/-- Theorem stating that for the given conditions, the time to return is 45 minutes --/
theorem time_to_return_is_45_minutes (journey : HikerJourney) 
  (h1 : journey.rate = 10)
  (h2 : journey.initialDistance = 2.5)
  (h3 : journey.totalDistance = 3.5) :
  timeToReturn journey = 45 := by
  sorry

end NUMINAMATH_CALUDE_time_to_return_is_45_minutes_l628_62843


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l628_62849

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_sixth : a 6 = Nat.factorial 9)
  (h_ninth : a 9 = Nat.factorial 10) :
  a 1 = (Nat.factorial 9) / (10 * Real.rpow 10 (1/3)) := by
  sorry

#check geometric_sequence_first_term

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l628_62849


namespace NUMINAMATH_CALUDE_fraction_simplification_l628_62835

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 7 + 3 = (26 + 4 * d) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l628_62835


namespace NUMINAMATH_CALUDE_max_value_at_neg_two_l628_62805

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_neg_two (c : ℝ) :
  (∀ x : ℝ, f c (-2) ≥ f c x) → c = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_neg_two_l628_62805


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l628_62855

theorem cubic_root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 504 * r + 1008 = 0) →
  (6 * s^3 + 504 * s + 1008 = 0) →
  (6 * t^3 + 504 * t + 1008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l628_62855


namespace NUMINAMATH_CALUDE_dress_price_after_discounts_l628_62817

theorem dress_price_after_discounts (d : ℝ) : 
  let initial_discount_rate : ℝ := 0.65
  let staff_discount_rate : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount_rate)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount_rate)
  final_price = d * 0.14 :=
by sorry

end NUMINAMATH_CALUDE_dress_price_after_discounts_l628_62817


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l628_62853

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the "in plane" relation
variable (inPlane : Line → Plane → Prop)

-- Define the intersection relation
variable (intersect : Line → Line → Prop)

-- Define our specific planes and lines
variable (α β : Plane)
variable (m n l₁ l₂ : Line)

-- State the theorem
theorem parallel_planes_condition
  (h1 : m ≠ n)
  (h2 : inPlane m α)
  (h3 : inPlane n α)
  (h4 : inPlane l₁ β)
  (h5 : inPlane l₂ β)
  (h6 : intersect l₁ l₂) :
  (lineParallel m l₁ ∧ lineParallel n l₂ → parallel α β) ∧
  ¬(parallel α β → lineParallel m l₁ ∧ lineParallel n l₂) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l628_62853


namespace NUMINAMATH_CALUDE_june_songs_total_l628_62836

def songs_in_june (vivian_daily : ℕ) (clara_difference : ℕ) (total_days : ℕ) (weekend_days : ℕ) : ℕ :=
  let weekdays := total_days - weekend_days
  let vivian_total := vivian_daily * weekdays
  let clara_daily := vivian_daily - clara_difference
  let clara_total := clara_daily * weekdays
  vivian_total + clara_total

theorem june_songs_total :
  songs_in_june 10 2 30 8 = 396 := by
  sorry

end NUMINAMATH_CALUDE_june_songs_total_l628_62836


namespace NUMINAMATH_CALUDE_unique_solution_x_three_halves_l628_62827

theorem unique_solution_x_three_halves :
  ∃! x : ℝ, ∀ y : ℝ, (y = (x^2 - 9) / (x - 3) ∧ y = 3*x) → x = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_x_three_halves_l628_62827


namespace NUMINAMATH_CALUDE_science_project_percentage_l628_62876

theorem science_project_percentage (total_pages math_pages remaining_pages : ℕ) 
  (h1 : total_pages = 120)
  (h2 : math_pages = 10)
  (h3 : remaining_pages = 80) :
  (total_pages - math_pages - remaining_pages) / total_pages * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_science_project_percentage_l628_62876


namespace NUMINAMATH_CALUDE_solve_equation_l628_62811

theorem solve_equation (x y : ℝ) (h1 : x^2 - x + 6 = y - 6) (h2 : x = -6) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l628_62811


namespace NUMINAMATH_CALUDE_expression_evaluation_l628_62866

theorem expression_evaluation : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l628_62866


namespace NUMINAMATH_CALUDE_sine_addition_formula_l628_62889

theorem sine_addition_formula (x y z : ℝ) :
  Real.sin (x + y) * Real.cos z + Real.cos (x + y) * Real.sin z = Real.sin (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_sine_addition_formula_l628_62889


namespace NUMINAMATH_CALUDE_negation_equivalence_l628_62839

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l628_62839


namespace NUMINAMATH_CALUDE_monotonic_decreasing_cubic_function_l628_62865

theorem monotonic_decreasing_cubic_function (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 1, 
    x < y → (a * x^3 - 3*x) > (a * y^3 - 3*y)) →
  0 < a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_cubic_function_l628_62865


namespace NUMINAMATH_CALUDE_prob_at_least_one_on_l628_62872

/-- The probability that at least one of three independent electronic components is on,
    given that each component has a probability of 1/2 of being on. -/
theorem prob_at_least_one_on (n : Nat) (p : ℝ) (h1 : n = 3) (h2 : p = 1 / 2) :
  1 - (1 - p) ^ n = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_on_l628_62872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_21st_term_l628_62897

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that the 21st term is -49. -/
theorem arithmetic_sequence_21st_term :
  let a : ℕ → ℤ := λ n => 11 + (n - 1) * (-3)
  a 21 = -49 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_21st_term_l628_62897


namespace NUMINAMATH_CALUDE_inequality_solution_set_l628_62868

-- Define the inequality
def inequality (x : ℝ) : Prop := (3*x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x ∈ solution_set ↔ inequality x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l628_62868


namespace NUMINAMATH_CALUDE_peach_difference_l628_62886

theorem peach_difference (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches > 2 * benjy_peaches →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches - 2 * benjy_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l628_62886


namespace NUMINAMATH_CALUDE_ball_distribution_after_four_rounds_l628_62848

/-- Represents the state of the game at any point -/
structure GameState :=
  (a b c d e : ℕ)

/-- Represents a single round of the game -/
def gameRound (s : GameState) : GameState :=
  let a' := if s.e < s.a then s.a - 2 else s.a
  let b' := if s.a < s.b then s.b - 2 else s.b
  let c' := if s.b < s.c then s.c - 2 else s.c
  let d' := if s.c < s.d then s.d - 2 else s.d
  let e' := if s.d < s.e then s.e - 2 else s.e
  ⟨a', b', c', d', e'⟩

/-- Represents the initial state of the game -/
def initialState : GameState := ⟨2, 4, 6, 8, 10⟩

/-- Represents the state after 4 rounds -/
def finalState : GameState := (gameRound ∘ gameRound ∘ gameRound ∘ gameRound) initialState

/-- The main theorem to be proved -/
theorem ball_distribution_after_four_rounds :
  finalState = ⟨6, 6, 6, 6, 6⟩ := by sorry

end NUMINAMATH_CALUDE_ball_distribution_after_four_rounds_l628_62848


namespace NUMINAMATH_CALUDE_f_property_l628_62851

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = -7 → f a b 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l628_62851


namespace NUMINAMATH_CALUDE_condition_equiv_range_l628_62881

/-- The set A in the real numbers -/
def A : Set ℝ := {x | -5 < x ∧ x < 4}

/-- The set B in the real numbers -/
def B : Set ℝ := {x | x < -6 ∨ x > 1}

/-- The set C in the real numbers, parameterized by m -/
def C (m : ℝ) : Set ℝ := {x | x < m}

/-- The theorem stating the equivalence of the conditions and the range of m -/
theorem condition_equiv_range :
  ∀ m : ℝ,
  (C m ⊇ (A ∩ B) ∧ C m ⊇ (Aᶜ ∩ Bᶜ)) ↔ m ∈ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_condition_equiv_range_l628_62881


namespace NUMINAMATH_CALUDE_cookie_cost_is_18_l628_62801

/-- The cost of each cookie Cora buys in April -/
def cookie_cost (cookies_per_day : ℕ) (days_in_april : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (cookies_per_day * days_in_april)

/-- Theorem stating that each cookie costs 18 dollars -/
theorem cookie_cost_is_18 :
  cookie_cost 3 30 1620 = 18 := by sorry

end NUMINAMATH_CALUDE_cookie_cost_is_18_l628_62801


namespace NUMINAMATH_CALUDE_lock_code_attempts_l628_62826

theorem lock_code_attempts (num_digits : ℕ) (code_length : ℕ) : 
  num_digits = 5 → code_length = 3 → num_digits ^ code_length - 1 = 124 := by
  sorry

#eval 5^3 - 1  -- This should output 124

end NUMINAMATH_CALUDE_lock_code_attempts_l628_62826


namespace NUMINAMATH_CALUDE_salt_mixture_percentage_l628_62823

/-- The percentage of salt in the initial solution -/
def P : ℝ := sorry

/-- The amount of initial solution in ounces -/
def initial_amount : ℝ := 40

/-- The amount of 60% solution added in ounces -/
def added_amount : ℝ := 40

/-- The percentage of salt in the added solution -/
def added_percentage : ℝ := 60

/-- The total amount of the resulting mixture in ounces -/
def total_amount : ℝ := initial_amount + added_amount

/-- The percentage of salt in the resulting mixture -/
def result_percentage : ℝ := 40

theorem salt_mixture_percentage :
  P = 20 ∧
  (P / 100 * initial_amount + added_percentage / 100 * added_amount) / total_amount * 100 = result_percentage :=
sorry

end NUMINAMATH_CALUDE_salt_mixture_percentage_l628_62823


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l628_62832

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 5 = 9 →
  a 2 + a 4 + a 6 = 15 →
  a 3 + a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l628_62832


namespace NUMINAMATH_CALUDE_worker_completion_time_l628_62857

/-- Given two workers P and Q, this theorem proves the time taken by Q to complete a task alone,
    given the time taken by P alone and the time taken by P and Q together. -/
theorem worker_completion_time (time_p : ℝ) (time_pq : ℝ) (time_q : ℝ) : 
  time_p = 15 → time_pq = 6 → time_q = 10 → 
  1 / time_pq = 1 / time_p + 1 / time_q :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l628_62857


namespace NUMINAMATH_CALUDE_wades_food_truck_l628_62893

/-- Wade's hot dog food truck problem -/
theorem wades_food_truck (tips_per_customer : ℚ) 
  (friday_customers sunday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  sunday_customers = 36 →
  total_tips = 296 →
  let saturday_customers := (total_tips - tips_per_customer * (friday_customers + sunday_customers)) / tips_per_customer
  (saturday_customers : ℚ) / friday_customers = 3 := by
  sorry

end NUMINAMATH_CALUDE_wades_food_truck_l628_62893


namespace NUMINAMATH_CALUDE_tea_set_problem_l628_62898

/-- Tea Set Problem -/
theorem tea_set_problem (cost_A cost_B : ℕ) 
  (h1 : cost_A + 2 * cost_B = 250)
  (h2 : 3 * cost_A + 4 * cost_B = 600)
  (h3 : ∀ a b : ℕ, a + b = 80 → 108 * a + 60 * b ≤ 6240)
  (h4 : ∀ a b : ℕ, a + b = 80 → 30 * a + 20 * b ≤ 1900)
  : ∃ a b : ℕ, a + b = 80 ∧ 30 * a + 20 * b = 1900 :=
sorry

end NUMINAMATH_CALUDE_tea_set_problem_l628_62898


namespace NUMINAMATH_CALUDE_max_points_is_four_l628_62804

/-- A configuration of points and associated real numbers satisfying the distance property -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximal number of points in a valid configuration is 4 -/
theorem max_points_is_four :
  (∃ (config : PointConfiguration), config.n = 4) ∧
  (∀ (config : PointConfiguration), config.n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_points_is_four_l628_62804


namespace NUMINAMATH_CALUDE_seven_people_round_table_l628_62894

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as the same arrangement -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 7 people
    around a round table is equal to 6! -/
theorem seven_people_round_table :
  roundTableArrangements 7 = Nat.factorial 6 := by sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l628_62894


namespace NUMINAMATH_CALUDE_gardener_tree_probability_l628_62875

theorem gardener_tree_probability (n_pine n_cedar n_fir : ℕ) 
  (h_pine : n_pine = 2)
  (h_cedar : n_cedar = 3)
  (h_fir : n_fir = 4) :
  let total_trees := n_pine + n_cedar + n_fir
  let non_fir_trees := n_pine + n_cedar
  let slots := non_fir_trees + 1
  let favorable_arrangements := Nat.choose slots n_fir
  let total_arrangements := Nat.choose total_trees n_fir
  let p := favorable_arrangements
  let q := total_arrangements
  p + q = 47 := by sorry

end NUMINAMATH_CALUDE_gardener_tree_probability_l628_62875


namespace NUMINAMATH_CALUDE_ninety_nine_times_one_hundred_one_l628_62852

theorem ninety_nine_times_one_hundred_one : 99 * 101 = 9999 := by
  sorry

end NUMINAMATH_CALUDE_ninety_nine_times_one_hundred_one_l628_62852


namespace NUMINAMATH_CALUDE_right_triangle_perfect_square_l628_62807

theorem right_triangle_perfect_square (a b c : ℕ) : 
  Prime a →
  a^2 + b^2 = c^2 →
  ∃ (n : ℕ), 2 * (a + b + 1) = n^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perfect_square_l628_62807


namespace NUMINAMATH_CALUDE_lattice_right_triangles_with_specific_incenter_l628_62822

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A right triangle with vertices O, A, and B, where O is the origin and the right angle -/
structure LatticeRightTriangle where
  A : LatticePoint
  B : LatticePoint

/-- The incenter of a right triangle -/
def incenter (t : LatticeRightTriangle) : ℚ × ℚ :=
  let a : ℚ := t.A.x
  let b : ℚ := t.B.y
  let c : ℚ := (a^2 + b^2).sqrt
  ((a + b - c) / 2, (a + b - c) / 2)

theorem lattice_right_triangles_with_specific_incenter :
  ∃ (n : ℕ), n > 0 ∧
  ∃ (triangles : Finset LatticeRightTriangle),
    triangles.card = n ∧
    ∀ t ∈ triangles, incenter t = (2015, 14105) := by
  sorry

end NUMINAMATH_CALUDE_lattice_right_triangles_with_specific_incenter_l628_62822


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l628_62895

def total_socks : ℕ := 12
def socks_per_color_a : ℕ := 3
def colors_with_three_socks : ℕ := 3
def colors_with_one_sock : ℕ := 2
def socks_drawn : ℕ := 5

def favorable_outcomes : ℕ := 
  (colors_with_three_socks.choose 2) * 
  (colors_with_one_sock.choose 1) * 
  (socks_per_color_a.choose 2) * 
  (socks_per_color_a.choose 2) * 
  1

def total_outcomes : ℕ := total_socks.choose socks_drawn

theorem sock_drawing_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 44 := by sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l628_62895


namespace NUMINAMATH_CALUDE_center_sum_is_six_l628_62892

/-- A circle in a shifted coordinate system -/
structure ShiftedCircle where
  -- The equation of the circle in the shifted system
  equation : (x y : ℝ) → Prop := fun x y => (x - 1)^2 + (y + 2)^2 = 4*x + 12*y + 6
  -- The shift of the coordinate system
  shift : ℝ × ℝ := (1, -2)

/-- The center of a circle in the standard coordinate system -/
def standardCenter (c : ShiftedCircle) : ℝ × ℝ := sorry

theorem center_sum_is_six (c : ShiftedCircle) : 
  let (h, k) := standardCenter c
  h + k = 6 := by sorry

end NUMINAMATH_CALUDE_center_sum_is_six_l628_62892


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l628_62879

theorem arithmetic_calculation : 15 * 35 - 15 * 5 + 25 * 15 = 825 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l628_62879


namespace NUMINAMATH_CALUDE_juan_birth_year_l628_62800

def first_btc_year : ℕ := 1990
def btc_frequency : ℕ := 2
def juan_age_at_fifth_btc : ℕ := 14

def btc_year (n : ℕ) : ℕ := first_btc_year + (n - 1) * btc_frequency

theorem juan_birth_year :
  first_btc_year = 1990 →
  btc_frequency = 2 →
  juan_age_at_fifth_btc = 14 →
  btc_year 5 - juan_age_at_fifth_btc = 1984 :=
by
  sorry

end NUMINAMATH_CALUDE_juan_birth_year_l628_62800


namespace NUMINAMATH_CALUDE_roden_fish_purchase_l628_62824

/-- Represents the number of fish bought in a single visit -/
structure FishPurchase where
  goldfish : ℕ
  bluefish : ℕ
  greenfish : ℕ

/-- Calculates the total number of fish bought during three visits -/
def totalFish (visit1 visit2 visit3 : FishPurchase) : ℕ :=
  visit1.goldfish + visit1.bluefish + visit1.greenfish +
  visit2.goldfish + visit2.bluefish + visit2.greenfish +
  visit3.goldfish + visit3.bluefish + visit3.greenfish

theorem roden_fish_purchase :
  let visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0 }
  let visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5 }
  let visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9 }
  totalFish visit1 visit2 visit3 = 68 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_purchase_l628_62824


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l628_62810

/-- Represents the seating arrangement problem --/
structure SeatingArrangement where
  front_seats : Nat
  back_seats : Nat
  people : Nat
  blocked_front : Nat

/-- Calculates the number of valid seating arrangements --/
def count_arrangements (s : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem seating_arrangement_count :
  let s : SeatingArrangement := {
    front_seats := 11,
    back_seats := 12,
    people := 2,
    blocked_front := 3
  }
  count_arrangements s = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l628_62810


namespace NUMINAMATH_CALUDE_apples_per_box_l628_62899

theorem apples_per_box 
  (apples_per_crate : ℕ) 
  (crates_delivered : ℕ) 
  (rotten_apples : ℕ) 
  (boxes_used : ℕ) 
  (h1 : apples_per_crate = 42)
  (h2 : crates_delivered = 12)
  (h3 : rotten_apples = 4)
  (h4 : boxes_used = 50)
  : (apples_per_crate * crates_delivered - rotten_apples) / boxes_used = 10 :=
by
  sorry

#check apples_per_box

end NUMINAMATH_CALUDE_apples_per_box_l628_62899


namespace NUMINAMATH_CALUDE_cone_volume_l628_62873

/-- Given a cone with base radius √3 cm and lateral area 6π cm², its volume is 3π cm³ -/
theorem cone_volume (r h : ℝ) : 
  r = Real.sqrt 3 → 
  2 * π * r * (Real.sqrt (h^2 + r^2)) = 6 * π → 
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l628_62873


namespace NUMINAMATH_CALUDE_solve_equation_l628_62840

-- Define a custom pair type for real numbers
structure RealPair :=
  (fst : ℝ)
  (snd : ℝ)

-- Define equality for RealPair
def realPairEq (a b : RealPair) : Prop :=
  a.fst = b.fst ∧ a.snd = b.snd

-- Define the ⊕ operation
def oplus (a b : RealPair) : RealPair :=
  ⟨a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst⟩

-- Theorem statement
theorem solve_equation (p q : ℝ) :
  oplus ⟨1, 2⟩ ⟨p, q⟩ = ⟨5, 0⟩ → realPairEq ⟨p, q⟩ ⟨1, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l628_62840


namespace NUMINAMATH_CALUDE_min_value_theorem_l628_62802

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y, 0 < y ∧ y < 1/2 ∧ 2/y + 9/(1-2*y) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l628_62802


namespace NUMINAMATH_CALUDE_no_perfect_squares_l628_62813

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- State the theorem
theorem no_perfect_squares : 
  ¬(is_perfect_square (factorial 100 * factorial 101)) ∧
  ¬(is_perfect_square (factorial 100 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 103)) ∧
  ¬(is_perfect_square (factorial 102 * factorial 103)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l628_62813


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l628_62882

def total_selling_price : ℕ := 18000
def total_length : ℕ := 300
def loss_per_metre : ℕ := 5

theorem cost_price_per_metre : 
  (total_selling_price / total_length) + loss_per_metre = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l628_62882


namespace NUMINAMATH_CALUDE_max_arrangement_length_l628_62833

def student_height : ℝ → Prop := λ h => h = 1.60 ∨ h = 1.22

def valid_arrangement (arrangement : List ℝ) : Prop :=
  (∀ i, i + 3 < arrangement.length → 
    (arrangement.take (i + 4)).sum / 4 > 1.50) ∧
  (∀ i, i + 6 < arrangement.length → 
    (arrangement.take (i + 7)).sum / 7 < 1.50)

theorem max_arrangement_length :
  ∃ (arrangement : List ℝ),
    arrangement.length = 9 ∧
    (∀ h ∈ arrangement, student_height h) ∧
    valid_arrangement arrangement ∧
    ∀ (longer_arrangement : List ℝ),
      longer_arrangement.length > 9 →
      (∀ h ∈ longer_arrangement, student_height h) →
      ¬(valid_arrangement longer_arrangement) :=
sorry

end NUMINAMATH_CALUDE_max_arrangement_length_l628_62833


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l628_62859

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l628_62859


namespace NUMINAMATH_CALUDE_intersection_sum_l628_62891

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l628_62891


namespace NUMINAMATH_CALUDE_part_one_part_two_l628_62818

-- Part 1
theorem part_one (x : ℝ) (a b : ℝ × ℝ) :
  a = (Real.sqrt 3 * Real.sin x, -1) →
  b = (Real.cos x, Real.sqrt 3) →
  ∃ (k : ℝ), a = k • b →
  (3 * Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) = -3 :=
sorry

-- Part 2
def f (x m : ℝ) (a b : ℝ × ℝ) : ℝ :=
  2 * ((a.1 + b.1) * b.1 + (a.2 + b.2) * b.2) - 2 * m^2 - 1

theorem part_two (m : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x m ((Real.sqrt 3 * Real.sin x, -1)) ((Real.cos x, m)) = 0) →
  m ∈ Set.Icc (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l628_62818


namespace NUMINAMATH_CALUDE_car_speed_problem_l628_62863

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_50 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_80 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A2 : ℝ := 80

theorem car_speed_problem :
  (speed_A1 * time_50 - speed_B * time_50 = speed_A2 * time_80 - speed_B * time_80) ∧
  speed_B = 35 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l628_62863


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l628_62888

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, (3*k - 2) * x^2 - 15*x + 13 ≠ 0) ∧
  (∀ k' : ℤ, k' < k → ∃ x : ℝ, (3*k' - 2) * x^2 - 15*x + 13 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l628_62888


namespace NUMINAMATH_CALUDE_seven_presenter_schedule_l628_62803

/-- The number of ways to schedule n presenters with one specific presenter following another --/
def schedule_presenters (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem: For 7 presenters, with one following another, there are 2520 ways to schedule --/
theorem seven_presenter_schedule :
  schedule_presenters 7 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_seven_presenter_schedule_l628_62803


namespace NUMINAMATH_CALUDE_number_2008_in_45th_group_l628_62877

/-- The sequence of arrays where the nth group has n numbers and the last number of the nth group is n(n+1) -/
def sequence_group (n : ℕ) : ℕ := n * (n + 1)

/-- The proposition that 2008 is in the 45th group of the sequence -/
theorem number_2008_in_45th_group :
  ∃ k : ℕ, k ≤ 45 ∧ 
  sequence_group 44 < 2008 ∧ 
  2008 ≤ sequence_group 45 :=
by sorry

end NUMINAMATH_CALUDE_number_2008_in_45th_group_l628_62877


namespace NUMINAMATH_CALUDE_gcd_lcm_45_150_l628_62830

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_150_l628_62830


namespace NUMINAMATH_CALUDE_smallest_math_club_size_l628_62867

theorem smallest_math_club_size : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (d : ℕ), d > 0 ∧ 
    (40 * n < 100 * d) ∧ 
    (100 * d < 50 * n)) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬(∃ (k : ℕ), k > 0 ∧ 
      (40 * m < 100 * k) ∧ 
      (100 * k < 50 * m))) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_math_club_size_l628_62867


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l628_62884

theorem quadratic_rewrite (k : ℝ) :
  let f := fun k : ℝ => 8 * k^2 - 6 * k + 16
  ∃ c r s : ℝ, (∀ k, f k = c * (k + r)^2 + s) ∧ s / r = -119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l628_62884


namespace NUMINAMATH_CALUDE_product_125_sum_31_l628_62869

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l628_62869


namespace NUMINAMATH_CALUDE_remainder_problem_l628_62856

theorem remainder_problem : (123456789012 : ℕ) % 252 = 84 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l628_62856


namespace NUMINAMATH_CALUDE_function_zero_at_seven_fifths_l628_62815

theorem function_zero_at_seven_fifths :
  let f : ℝ → ℝ := λ x ↦ 5 * x - 7
  f (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_at_seven_fifths_l628_62815


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_one_over_sqrt_two_l628_62870

/-- A right-angled triangle with its height and inscribed circles -/
structure RightTriangleWithInscribedCircles where
  /-- The original right-angled triangle -/
  originalTriangle : Set (ℝ × ℝ)
  /-- The two triangles formed by the height -/
  subTriangle1 : Set (ℝ × ℝ)
  subTriangle2 : Set (ℝ × ℝ)
  /-- The center of the inscribed circle of subTriangle1 -/
  center1 : ℝ × ℝ
  /-- The center of the inscribed circle of subTriangle2 -/
  center2 : ℝ × ℝ
  /-- The distance between center1 and center2 is 1 -/
  centers_distance : Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) = 1
  /-- The height divides the original triangle into subTriangle1 and subTriangle2 -/
  height_divides : originalTriangle = subTriangle1 ∪ subTriangle2
  /-- The original triangle is right-angled -/
  is_right_angled : ∃ (a b c : ℝ × ℝ), a ∈ originalTriangle ∧ b ∈ originalTriangle ∧ c ∈ originalTriangle ∧
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The radius of the inscribed circle of the original triangle -/
def inscribed_circle_radius (t : RightTriangleWithInscribedCircles) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle of the original triangle is 1/√2 -/
theorem inscribed_circle_radius_is_one_over_sqrt_two (t : RightTriangleWithInscribedCircles) :
  inscribed_circle_radius t = 1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_one_over_sqrt_two_l628_62870


namespace NUMINAMATH_CALUDE_actual_sampling_method_is_other_l628_62850

/-- Represents the sampling method used in the survey --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | Other

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  location : String
  selection : String
  endCondition : String

/-- The actual sampling process used in the survey --/
def actualSamplingProcess : SamplingProcess :=
  { location := "shopping mall entrance",
    selection := "randomly selected individuals",
    endCondition := "until predetermined number of respondents reached" }

/-- Theorem stating that the actual sampling method is not one of the three standard methods --/
theorem actual_sampling_method_is_other (sm : SamplingMethod) 
  (h : sm = SamplingMethod.SimpleRandom ∨ 
       sm = SamplingMethod.Stratified ∨ 
       sm = SamplingMethod.Systematic) : 
  sm ≠ SamplingMethod.Other → False := by
  sorry

end NUMINAMATH_CALUDE_actual_sampling_method_is_other_l628_62850


namespace NUMINAMATH_CALUDE_unique_function_existence_l628_62846

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_existence :
  ∃! f : RealFunction, ∀ x y : ℝ, f (x + f y) = x + y + 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l628_62846


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l628_62821

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l628_62821


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_KLMN_l628_62828

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define points K, L, N
structure Points (t : Triangle) :=
  (bk : ℝ)
  (bl : ℝ)
  (an : ℝ)

-- Define the quadrilateral KLMN
def quadrilateral_area (t : Triangle) (p : Points t) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral_KLMN :
  let t : Triangle := { a := 13, b := 14, c := 15 }
  let p : Points t := { bk := 14/13, bl := 1, an := 10 }
  quadrilateral_area t p = 36503/1183 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_KLMN_l628_62828


namespace NUMINAMATH_CALUDE_larger_number_proof_l628_62887

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 8 * S + 15) :
  L = 1557 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l628_62887


namespace NUMINAMATH_CALUDE_emails_morning_evening_l628_62862

def morning_emails : ℕ := 3
def evening_emails : ℕ := 8

theorem emails_morning_evening : 
  morning_emails + evening_emails = 11 :=
by sorry

end NUMINAMATH_CALUDE_emails_morning_evening_l628_62862


namespace NUMINAMATH_CALUDE_final_symbol_invariant_l628_62842

/-- Represents the state of the blackboard -/
structure BlackboardState where
  minus_count : Nat
  total_count : Nat

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_same_plus
  | erase_same_minus
  | erase_different

/-- Applies an operation to the blackboard state -/
def apply_operation (state : BlackboardState) (op : Operation) : BlackboardState :=
  match op with
  | Operation.erase_same_plus => ⟨state.minus_count, state.total_count - 1⟩
  | Operation.erase_same_minus => ⟨state.minus_count - 2, state.total_count - 1⟩
  | Operation.erase_different => ⟨state.minus_count, state.total_count - 1⟩

/-- The main theorem stating that the final symbol is determined by the initial parity of minus signs -/
theorem final_symbol_invariant (initial_state : BlackboardState)
  (h_initial : initial_state.total_count = 1967)
  (h_valid : initial_state.minus_count ≤ initial_state.total_count) :
  ∃ (final_symbol : Bool),
    ∀ (ops : List Operation),
      (ops.foldl apply_operation initial_state).total_count = 1 →
      final_symbol = ((ops.foldl apply_operation initial_state).minus_count % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_final_symbol_invariant_l628_62842


namespace NUMINAMATH_CALUDE_coefficient_of_x_plus_two_to_ten_l628_62834

theorem coefficient_of_x_plus_two_to_ten (x : ℝ) :
  ∃ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
    (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
      a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + 
      a₈*(x + 2)^8 + a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11 ∧
    a₁₀ = -11 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_plus_two_to_ten_l628_62834


namespace NUMINAMATH_CALUDE_exist_ten_special_integers_l628_62841

theorem exist_ten_special_integers : 
  ∃ (a : Fin 10 → ℕ+), 
    (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧ 
    (∀ i j, (a i)^2 ∣ a j) := by
  sorry

end NUMINAMATH_CALUDE_exist_ten_special_integers_l628_62841


namespace NUMINAMATH_CALUDE_binomial_10_2_l628_62814

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l628_62814


namespace NUMINAMATH_CALUDE_marys_income_percentage_l628_62871

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 0.8 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l628_62871


namespace NUMINAMATH_CALUDE_expression_evaluation_l628_62806

theorem expression_evaluation :
  ∃ (m : ℕ+), (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l628_62806


namespace NUMINAMATH_CALUDE_zoom_download_time_l628_62847

theorem zoom_download_time (total_time audio_glitch_time video_glitch_time : ℕ)
  (h_total : total_time = 82)
  (h_audio : audio_glitch_time = 2 * 4)
  (h_video : video_glitch_time = 6)
  (h_glitch_ratio : 2 * (audio_glitch_time + video_glitch_time) = total_time - (audio_glitch_time + video_glitch_time) - 40) :
  let mac_download_time := (total_time - (audio_glitch_time + video_glitch_time) - 2 * (audio_glitch_time + video_glitch_time)) / 4
  mac_download_time = 10 := by sorry

end NUMINAMATH_CALUDE_zoom_download_time_l628_62847


namespace NUMINAMATH_CALUDE_parabola_equation_l628_62838

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

/-- Definition of the left vertex of the hyperbola -/
def left_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y = 0

/-- Definition of a parabola passing through a point -/
def parabola_through_point (eq : ℝ → ℝ → Prop) (px py : ℝ) : Prop :=
  eq px py

/-- Theorem stating the standard equation of the parabola -/
theorem parabola_equation (f : ℝ → ℝ → Prop) (fx fy : ℝ) :
  left_vertex fx fy →
  parabola_through_point f 2 (-4) →
  (∀ x y, f x y ↔ y^2 = 8*x) ∨ (∀ x y, f x y ↔ x^2 = -y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l628_62838


namespace NUMINAMATH_CALUDE_f_properties_l628_62819

noncomputable def f (x : ℝ) := 2 * abs (Real.sin x + Real.cos x) - Real.sin (2 * x)

theorem f_properties :
  (∀ x, f (π / 2 - x) = f x) ∧
  (∀ x, f x ≥ 1) ∧
  (∀ x y, π / 4 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l628_62819


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_minus_two_alpha_l628_62864

theorem cos_two_thirds_pi_minus_two_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) = Real.sqrt 6 / 3) : 
  Real.cos (2 * π / 3 - 2 * α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_minus_two_alpha_l628_62864


namespace NUMINAMATH_CALUDE_average_speed_not_necessarily_five_l628_62880

/-- A pedestrian's walk with varying speeds over 2.5 hours -/
structure PedestrianWalk where
  duration : ℝ
  hourly_distance : ℝ
  average_speed : ℝ

/-- Axiom: The pedestrian walks for 2.5 hours -/
axiom walk_duration : ∀ (w : PedestrianWalk), w.duration = 2.5

/-- Axiom: The pedestrian covers 5 km in any one-hour interval -/
axiom hourly_distance : ∀ (w : PedestrianWalk), w.hourly_distance = 5

/-- Theorem: The average speed for the entire journey is not necessarily 5 km per hour -/
theorem average_speed_not_necessarily_five :
  ∃ (w : PedestrianWalk), w.average_speed ≠ 5 := by
  sorry


end NUMINAMATH_CALUDE_average_speed_not_necessarily_five_l628_62880


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l628_62885

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l628_62885


namespace NUMINAMATH_CALUDE_square_perimeter_l628_62809

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) :
  4 * side = 120 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l628_62809


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l628_62837

theorem quadratic_function_inequality (a b c : ℝ) (h1 : c > b) (h2 : b > a) 
  (h3 : a * 1^2 + 2 * b * 1 + c = 0) 
  (h4 : ∃ x, a * x^2 + 2 * b * x + c = -a) : 
  0 ≤ b / a ∧ b / a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l628_62837


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l628_62861

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
    a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
    a + b + c = 2*c          -- perimeter equals twice the hypotenuse
    := by sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l628_62861


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l628_62860

/-- Given a square with side z containing a smaller square with side w,
    prove that the perimeter of a rectangle formed by the remaining area is 2z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hw_lt_z : w < z) :
  2 * w + 2 * (z - w) = 2 * z := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l628_62860


namespace NUMINAMATH_CALUDE_cubic_root_property_l628_62883

-- Define the cubic polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the roots and their properties
theorem cubic_root_property (x₁ x₂ x₃ : ℝ) 
  (h1 : f x₁ = 0) (h2 : f x₂ = 0) (h3 : f x₃ = 0)
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  x₃^2 - x₂^2 = x₃ - x₁ := by
sorry

end NUMINAMATH_CALUDE_cubic_root_property_l628_62883


namespace NUMINAMATH_CALUDE_pentagon_side_length_l628_62820

/-- Given a triangle with all sides of length 20/9 cm and a pentagon with the same perimeter
    and all sides of equal length, the length of one side of the pentagon is 4/3 cm. -/
theorem pentagon_side_length (triangle_side : ℚ) (pentagon_side : ℚ) :
  triangle_side = 20 / 9 →
  3 * triangle_side = 5 * pentagon_side →
  pentagon_side = 4 / 3 := by
  sorry

#eval (4 : ℚ) / 3  -- Expected output: 4/3

end NUMINAMATH_CALUDE_pentagon_side_length_l628_62820


namespace NUMINAMATH_CALUDE_alpha_beta_values_l628_62808

theorem alpha_beta_values (n k : ℤ) :
  let α : ℝ := π / 4 + 2 * π * (n : ℝ)
  let β : ℝ := π / 3 + 2 * π * (k : ℝ)
  (∃ m : ℤ, α = π / 4 + 2 * π * (m : ℝ)) ∧
  (∃ l : ℤ, β = π / 3 + 2 * π * (l : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_values_l628_62808


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l628_62854

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  2 * r₁^2 + 5 * r₁ = 12 ∧
  2 * r₂^2 + 5 * r₂ = 12 ∧
  abs (r₁ - r₂) = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l628_62854


namespace NUMINAMATH_CALUDE_cost_increase_doubles_b_l628_62874

/-- The cost function for a given parameter b and coefficient t -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that if the new cost is 1600% of the original cost,
    then the new value of b is 2 times the original value -/
theorem cost_increase_doubles_b (t : ℝ) (b₁ b₂ : ℝ) (h : t > 0) :
  cost t b₂ = 16 * cost t b₁ → b₂ = 2 * b₁ := by
  sorry


end NUMINAMATH_CALUDE_cost_increase_doubles_b_l628_62874


namespace NUMINAMATH_CALUDE_circle_sum_is_twenty_l628_62890

def CircleSum (digits : Finset ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d e f x : ℕ),
    digits = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    5 ∈ digits ∧
    2 ∈ digits ∧
    x + a + b + 5 = sum ∧
    x + e + f + 2 = sum ∧
    5 + c + d + 2 = sum

theorem circle_sum_is_twenty :
  ∃ (digits : Finset ℕ) (sum : ℕ), CircleSum digits sum ∧ sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_is_twenty_l628_62890


namespace NUMINAMATH_CALUDE_square_plus_one_nonnegative_l628_62858

theorem square_plus_one_nonnegative (m : ℝ) : m^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_nonnegative_l628_62858


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l628_62812

theorem greatest_four_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≤ 9996 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l628_62812


namespace NUMINAMATH_CALUDE_class_size_from_ratio_and_red_hair_count_l628_62816

/-- Represents the number of children with each hair color in the ratio --/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Calculates the total parts in the ratio --/
def totalParts (ratio : HairColorRatio) : ℕ :=
  ratio.red + ratio.blonde + ratio.black

/-- Theorem: Given the hair color ratio and number of red-haired children, 
    prove the total number of children in the class --/
theorem class_size_from_ratio_and_red_hair_count 
  (ratio : HairColorRatio) 
  (red_hair_count : ℕ) 
  (h1 : ratio.red = 3) 
  (h2 : ratio.blonde = 6) 
  (h3 : ratio.black = 7) 
  (h4 : red_hair_count = 9) : 
  (red_hair_count * totalParts ratio) / ratio.red = 48 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_ratio_and_red_hair_count_l628_62816


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l628_62825

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 150)
  (h2 : rejected_meters = 15) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l628_62825


namespace NUMINAMATH_CALUDE_sum_multiple_of_three_l628_62845

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_of_three_l628_62845


namespace NUMINAMATH_CALUDE_log_3125_base_5_between_consecutive_integers_l628_62829

theorem log_3125_base_5_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 3125 / Real.log 5 ∧ Real.log 3125 / Real.log 5 < (d : ℝ) ∧ c + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_3125_base_5_between_consecutive_integers_l628_62829


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l628_62896

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℚ, 16 * x^2 + 40 * x + 18 = (a * x + b)^2 + c) →
  a * b = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l628_62896


namespace NUMINAMATH_CALUDE_horseshoe_production_theorem_l628_62831

/-- Represents the manufacturing and sales scenario for horseshoes --/
structure HorseshoeScenario where
  initialOutlay : ℕ
  costPerSet : ℕ
  sellingPricePerSet : ℕ
  profit : ℕ

/-- Calculates the number of sets of horseshoes produced and sold --/
def setsProducedAndSold (scenario : HorseshoeScenario) : ℕ :=
  (scenario.profit + scenario.initialOutlay) / (scenario.sellingPricePerSet - scenario.costPerSet)

/-- Theorem stating that the number of sets produced and sold is 500 --/
theorem horseshoe_production_theorem (scenario : HorseshoeScenario) 
  (h1 : scenario.initialOutlay = 10000)
  (h2 : scenario.costPerSet = 20)
  (h3 : scenario.sellingPricePerSet = 50)
  (h4 : scenario.profit = 5000) :
  setsProducedAndSold scenario = 500 := by
  sorry

#eval setsProducedAndSold { initialOutlay := 10000, costPerSet := 20, sellingPricePerSet := 50, profit := 5000 }

end NUMINAMATH_CALUDE_horseshoe_production_theorem_l628_62831
