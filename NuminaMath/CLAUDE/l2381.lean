import Mathlib

namespace root_square_value_l2381_238167

theorem root_square_value (x₁ x₂ : ℂ) : 
  x₁ ≠ x₂ →
  (x₁ - 1)^2 = -3 →
  (x₂ - 1)^2 = -3 →
  x₁ = 1 - Complex.I * Real.sqrt 3 →
  x₂^2 = -2 + 2 * Complex.I * Real.sqrt 3 := by
sorry

end root_square_value_l2381_238167


namespace faye_pencils_l2381_238141

/-- The number of pencils Faye has in all sets -/
def total_pencils (rows_per_set : ℕ) (pencils_per_row : ℕ) (num_sets : ℕ) : ℕ :=
  rows_per_set * pencils_per_row * num_sets

/-- Theorem stating the total number of pencils Faye has -/
theorem faye_pencils :
  total_pencils 14 11 3 = 462 := by
  sorry

end faye_pencils_l2381_238141


namespace disk_color_difference_l2381_238136

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let ratio_sum := blue_ratio + yellow_ratio + green_ratio
  let blue_count := (blue_ratio * total) / ratio_sum
  let green_count := (green_ratio * total) / ratio_sum
  green_count - blue_count = 35 := by
sorry

end disk_color_difference_l2381_238136


namespace f_of_5_eq_19_l2381_238171

/-- Given f(x) = (7x + 3) / (x - 3), prove that f(5) = 19 -/
theorem f_of_5_eq_19 : 
  let f : ℝ → ℝ := λ x ↦ (7 * x + 3) / (x - 3)
  f 5 = 19 := by
  sorry

end f_of_5_eq_19_l2381_238171


namespace may_friday_to_monday_l2381_238138

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in May -/
structure DayInMay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The function that determines the day of the week for a given day in May -/
def dayOfWeekInMay (d : Nat) : DayOfWeek :=
  sorry

theorem may_friday_to_monday (r n : Nat) 
  (h1 : dayOfWeekInMay r = DayOfWeek.Friday)
  (h2 : dayOfWeekInMay n = DayOfWeek.Monday)
  (h3 : 15 < n)
  (h4 : n < 25) :
  n = 20 := by
  sorry

end may_friday_to_monday_l2381_238138


namespace product_sum_theorem_l2381_238108

theorem product_sum_theorem (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120 →
  p + q + r + s + t = 32 := by
sorry

end product_sum_theorem_l2381_238108


namespace complex_equation_solution_l2381_238105

theorem complex_equation_solution (z : ℂ) : (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = -1/5 - 3/5 * Complex.I := by
  sorry

end complex_equation_solution_l2381_238105


namespace initial_number_proof_l2381_238157

theorem initial_number_proof (x : ℕ) : x + 17 = 29 → x = 12 := by
  sorry

end initial_number_proof_l2381_238157


namespace regular_polygon_exterior_angle_18_deg_has_20_sides_l2381_238149

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ, n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by sorry

end regular_polygon_exterior_angle_18_deg_has_20_sides_l2381_238149


namespace fruit_basket_problem_l2381_238133

theorem fruit_basket_problem (total_fruits : ℕ) 
  (mangoes pears pawpaws kiwis lemons : ℕ) : 
  total_fruits = 58 →
  mangoes = 18 →
  pears = 10 →
  pawpaws = 12 →
  kiwis = lemons →
  total_fruits = mangoes + pears + pawpaws + kiwis + lemons →
  lemons = 9 := by
sorry

end fruit_basket_problem_l2381_238133


namespace angle_problem_l2381_238183

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  sorry

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  sorry

/-- Equality of two angles -/
def Angle.eq (a b : Angle) : Prop :=
  sorry

theorem angle_problem (x y : Angle) :
  Angle.add x y = Angle.mk 67 56 →
  Angle.sub x y = Angle.mk 12 40 →
  Angle.eq x (Angle.mk 40 18) ∧ Angle.eq y (Angle.mk 27 38) :=
by
  sorry

end angle_problem_l2381_238183


namespace quadratic_two_roots_condition_l2381_238190

/-- 
For a quadratic equation x^2 - 2x + k = 0 to have two real roots, 
k must satisfy k ≤ 1
-/
theorem quadratic_two_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) →
  k ≤ 1 :=
by sorry

end quadratic_two_roots_condition_l2381_238190


namespace square_of_negative_two_x_squared_l2381_238150

theorem square_of_negative_two_x_squared (x : ℝ) : (-2 * x^2)^2 = 4 * x^4 := by
  sorry

end square_of_negative_two_x_squared_l2381_238150


namespace store_visitors_l2381_238139

theorem store_visitors (first_hour_left second_hour_in second_hour_out final_count : ℕ) :
  first_hour_left = 27 →
  second_hour_in = 18 →
  second_hour_out = 9 →
  final_count = 76 →
  ∃ first_hour_in : ℕ, first_hour_in = 94 ∧
    final_count = first_hour_in - first_hour_left + second_hour_in - second_hour_out :=
by sorry

end store_visitors_l2381_238139


namespace product_without_linear_term_l2381_238148

theorem product_without_linear_term (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 8) = a * x^2 + b) → m = -8 := by
  sorry

end product_without_linear_term_l2381_238148


namespace sphere_area_is_14pi_l2381_238160

/-- A cuboid with vertices on a sphere's surface -/
structure CuboidOnSphere where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  vertices_on_sphere : Bool

/-- The surface area of a sphere containing a cuboid -/
def sphere_surface_area (c : CuboidOnSphere) : ℝ := sorry

/-- Theorem: The surface area of the sphere is 14π -/
theorem sphere_area_is_14pi (c : CuboidOnSphere) 
  (h1 : c.edge1 = 1) 
  (h2 : c.edge2 = 2) 
  (h3 : c.edge3 = 3) 
  (h4 : c.vertices_on_sphere = true) : 
  sphere_surface_area c = 14 * Real.pi := by sorry

end sphere_area_is_14pi_l2381_238160


namespace unique_input_for_542_l2381_238121

def machine_operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 * n else 3 * n + 2

def iterate_machine (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => machine_operation (iterate_machine n k)

theorem unique_input_for_542 :
  ∃! n : ℕ, n > 0 ∧ iterate_machine n 5 = 542 :=
by
  -- The proof would go here
  sorry

#eval iterate_machine 112500 5  -- Should output 542

end unique_input_for_542_l2381_238121


namespace equation_roots_l2381_238184

theorem equation_roots (m : ℝ) : 
  (∃! x : ℝ, (m - 2) * x^2 - 2 * (m - 1) * x + m = 0) → 
  (∃ x : ℝ, ∀ y : ℝ, m * y^2 - (m + 2) * y + (4 - m) = 0 ↔ y = x) := by
  sorry

end equation_roots_l2381_238184


namespace no_definitive_conclusion_l2381_238110

-- Define the sets
variable (Beta Zeta Yota : Set α)

-- Define the hypotheses
variable (h1 : ∃ x, x ∈ Beta ∧ x ∉ Zeta)
variable (h2 : Zeta ⊆ Yota)

-- Define the statements that cannot be conclusively proven
def statement_A := ∃ x, x ∈ Beta ∧ x ∉ Yota
def statement_B := Beta ⊆ Yota
def statement_C := Beta ∩ Yota = ∅
def statement_D := ∃ x, x ∈ Beta ∧ x ∈ Yota

-- Theorem stating that none of the statements can be definitively concluded
theorem no_definitive_conclusion :
  ¬(statement_A Beta Yota ∨ statement_B Beta Yota ∨ statement_C Beta Yota ∨ statement_D Beta Yota) :=
sorry

end no_definitive_conclusion_l2381_238110


namespace ball_probabilities_l2381_238122

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end ball_probabilities_l2381_238122


namespace curve_C_polar_equation_l2381_238127

/-- Given a curve C with parametric equations x = 1 + cos α and y = sin α, 
    its polar equation is ρ = 2cos θ -/
theorem curve_C_polar_equation (α θ : Real) (ρ x y : Real) :
  (x = 1 + Real.cos α ∧ y = Real.sin α) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.cos θ := by
  sorry

end curve_C_polar_equation_l2381_238127


namespace parallel_vectors_k_value_l2381_238179

-- Define the points A and B
def A : ℝ × ℝ := (2, -2)
def B : ℝ × ℝ := (4, 3)

-- Define vector a as a function of k
def a (k : ℝ) : ℝ × ℝ := (2*k - 1, 7)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem parallel_vectors_k_value : 
  ∃ (c : ℝ), c ≠ 0 ∧ a (19/10) = (c * AB.1, c * AB.2) :=
sorry

end parallel_vectors_k_value_l2381_238179


namespace max_prism_plane_intersections_l2381_238174

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Represents the base of the prism
  height : ℝ           -- Represents the height of the prism

/-- A plane in three-dimensional space. -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  d : ℝ                -- Distance from the origin

/-- Represents the number of edges a plane intersects with a prism. -/
def intersectionCount (prism : Prism) (plane : Plane) : ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The maximum number of edges a plane can intersect in a prism is 8. -/
theorem max_prism_plane_intersections (prism : Prism) :
  ∀ plane : Plane, intersectionCount prism plane ≤ 8 :=
sorry

end max_prism_plane_intersections_l2381_238174


namespace arun_remaining_work_days_arun_remaining_work_days_proof_l2381_238195

-- Define the work rates and time
def arun_tarun_rate : ℚ := 1 / 10
def arun_rate : ℚ := 1 / 60
def initial_work_days : ℕ := 4
def total_work : ℚ := 1

-- Theorem statement
theorem arun_remaining_work_days : ℕ :=
  let remaining_work : ℚ := total_work - (arun_tarun_rate * initial_work_days)
  let arun_remaining_days : ℚ := remaining_work / arun_rate
  36

-- Proof
theorem arun_remaining_work_days_proof :
  arun_remaining_work_days = 36 := by
  sorry

end arun_remaining_work_days_arun_remaining_work_days_proof_l2381_238195


namespace max_gcd_11n_plus_3_6n_plus_1_l2381_238166

theorem max_gcd_11n_plus_3_6n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ gcd (11 * k + 3) (6 * k + 1) = 7 ∧
  ∀ (n : ℕ), n > 0 → gcd (11 * n + 3) (6 * n + 1) ≤ 7 :=
sorry

end max_gcd_11n_plus_3_6n_plus_1_l2381_238166


namespace sum_of_roots_when_product_is_24_l2381_238147

theorem sum_of_roots_when_product_is_24 (x₁ x₂ : ℝ) :
  (x₁ + 3) * (x₁ - 4) = 24 →
  (x₂ + 3) * (x₂ - 4) = 24 →
  x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_when_product_is_24_l2381_238147


namespace solution_difference_l2381_238180

theorem solution_difference : ∃ (a b : ℝ), 
  (∀ x : ℝ, (3*x - 9) / (x^2 + x - 6) = x + 1 ↔ (x = a ∨ x = b)) ∧ 
  a > b ∧ 
  a - b = 4 := by
sorry

end solution_difference_l2381_238180


namespace salaria_tree_count_l2381_238102

/-- Represents the types of orange trees --/
inductive TreeType
| A
| B

/-- Calculates the number of good oranges per tree per month --/
def goodOrangesPerTree (t : TreeType) : ℚ :=
  match t with
  | TreeType.A => 10 * (60 / 100)
  | TreeType.B => 15 * (1 / 3)

/-- Calculates the average number of good oranges per tree per month --/
def avgGoodOrangesPerTree : ℚ :=
  (goodOrangesPerTree TreeType.A + goodOrangesPerTree TreeType.B) / 2

/-- The total number of good oranges Salaria gets per month --/
def totalGoodOranges : ℚ := 55

/-- Theorem stating that the total number of trees Salaria has is 10 --/
theorem salaria_tree_count :
  totalGoodOranges / avgGoodOrangesPerTree = 10 := by
  sorry


end salaria_tree_count_l2381_238102


namespace vector_simplification_1_vector_simplification_2_l2381_238131

variable {V : Type*} [AddCommGroup V]

-- Define vectors
variable (A B C D E O : V)

-- Define the vector operations
def vec (X Y : V) := Y - X

-- Theorem statements
theorem vector_simplification_1 :
  (vec B A - vec B C) - (vec E D - vec E C) = vec D A := by sorry

theorem vector_simplification_2 :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end vector_simplification_1_vector_simplification_2_l2381_238131


namespace remainder_problem_l2381_238151

theorem remainder_problem (m : ℤ) (h : m % 24 = 23) : m % 288 = 23 := by
  sorry

end remainder_problem_l2381_238151


namespace melody_reading_pages_l2381_238137

theorem melody_reading_pages (science civics chinese : ℕ) (total_tomorrow : ℕ) (english : ℕ) : 
  science = 16 → 
  civics = 8 → 
  chinese = 12 → 
  total_tomorrow = 14 → 
  (english / 4 + science / 4 + civics / 4 + chinese / 4 : ℚ) = total_tomorrow → 
  english = 20 := by
sorry

end melody_reading_pages_l2381_238137


namespace women_who_left_l2381_238114

theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) (final_men : ℕ) (final_women : ℕ) :
  initial_men * 5 = initial_women * 4 →
  final_men = initial_men + 2 →
  final_men = 14 →
  final_women = 24 →
  final_women = 2 * (initial_women - (initial_women - final_women / 2)) →
  initial_women - final_women / 2 = 3 :=
by sorry

end women_who_left_l2381_238114


namespace split_bill_example_l2381_238188

/-- Calculates the amount each person should pay when splitting a bill equally -/
def split_bill (num_people : ℕ) (num_bread : ℕ) (bread_price : ℕ) (num_hotteok : ℕ) (hotteok_price : ℕ) : ℕ :=
  ((num_bread * bread_price + num_hotteok * hotteok_price) / num_people)

/-- Theorem stating that given the conditions, each person should pay 1650 won -/
theorem split_bill_example : split_bill 4 5 200 7 800 = 1650 := by
  sorry

end split_bill_example_l2381_238188


namespace min_value_implies_a_range_l2381_238125

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

/-- Theorem stating that if f(2) is the minimum value of f(x), then a ∈ [2, 5] -/
theorem min_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 2) → 2 ≤ a ∧ a ≤ 5 := by
  sorry


end min_value_implies_a_range_l2381_238125


namespace linear_function_intersection_l2381_238175

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 3 = 0 ∧ x^2 = 36) → (k = 1/2 ∨ k = -1/2) := by
  sorry

end linear_function_intersection_l2381_238175


namespace continuous_function_on_T_has_fixed_point_l2381_238101

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ) (q : ℚ), t ∈ Set.Icc 0 1 ∧ p = (t * q, 1 - t)}

-- State the theorem
theorem continuous_function_on_T_has_fixed_point
  (f : T → T) (hf : Continuous f) :
  ∃ x : T, f x = x := by
  sorry

end continuous_function_on_T_has_fixed_point_l2381_238101


namespace triangle_reciprocal_sum_l2381_238173

/-- Given a triangle with sides a, b, c, semiperimeter p, inradius r, and circumradius R,
    prove that 1/ab + 1/bc + 1/ac = 1/(2rR) -/
theorem triangle_reciprocal_sum (a b c p r R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ r > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * p))
  (h_circumradius : R = (a * b * c) / (4 * (p - a) * (p - b) * (p - c))) :
  1 / (a * b) + 1 / (b * c) + 1 / (a * c) = 1 / (2 * r * R) := by
  sorry

end triangle_reciprocal_sum_l2381_238173


namespace total_spent_on_decks_l2381_238155

/-- The cost of a trick deck in dollars -/
def deck_cost : ℝ := 8

/-- The discount rate for buying 5 or more decks -/
def discount_rate : ℝ := 0.1

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Alice bought -/
def alice_decks : ℕ := 4

/-- The number of decks Bob bought -/
def bob_decks : ℕ := 3

/-- The minimum number of decks to qualify for a discount -/
def discount_threshold : ℕ := 5

/-- Function to calculate the cost of decks with potential discount -/
def calculate_cost (num_decks : ℕ) : ℝ :=
  let base_cost := (num_decks : ℝ) * deck_cost
  if num_decks ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the total amount spent on trick decks -/
theorem total_spent_on_decks : 
  calculate_cost victor_decks + calculate_cost alice_decks + calculate_cost bob_decks = 99.20 := by
  sorry

end total_spent_on_decks_l2381_238155


namespace train_length_calculation_l2381_238132

/-- The length of a train given its speed and time to cross a pole. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train with speed 53.99999999999999 m/s that crosses a pole in 20 seconds has a length of 1080 meters. -/
theorem train_length_calculation :
  let speed : ℝ := 53.99999999999999
  let time : ℝ := 20
  train_length speed time = 1080 := by
  sorry

end train_length_calculation_l2381_238132


namespace screen_area_difference_l2381_238193

theorem screen_area_difference :
  let square_area (diagonal : ℝ) := diagonal^2 / 2
  (square_area 19 - square_area 17) = 36 := by sorry

end screen_area_difference_l2381_238193


namespace students_without_A_l2381_238185

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_history : history = 12)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 16 := by
  sorry

end students_without_A_l2381_238185


namespace time_after_56_hours_l2381_238161

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Adds hours to a given time -/
def addHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + h * 60
  { hour := (totalMinutes / 60) % 24, minute := totalMinutes % 60 }

theorem time_after_56_hours (start : Time) (h : Nat) :
  start = { hour := 9, minute := 4 } →
  h = 56 →
  addHours start h = { hour := 17, minute := 4 } := by
  sorry

end time_after_56_hours_l2381_238161


namespace coffee_shop_optimal_price_l2381_238199

/-- Profit function for the coffee shop -/
def profit (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The constraint on the price -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 19

theorem coffee_shop_optimal_price :
  ∃ (p : ℝ), price_constraint p ∧ 
  ∀ (q : ℝ), price_constraint q → profit p ≥ profit q ∧
  p = optimal_price :=
sorry

end coffee_shop_optimal_price_l2381_238199


namespace max_cards_purchasable_l2381_238189

theorem max_cards_purchasable (budget : ℚ) (card_cost : ℚ) (h1 : budget = 15/2) (h2 : card_cost = 17/20) :
  ⌊budget / card_cost⌋ = 8 := by
  sorry

end max_cards_purchasable_l2381_238189


namespace inequality_proof_l2381_238170

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) ≤ 1 ∧
  ((a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_proof_l2381_238170


namespace imaginary_part_of_pure_imaginary_z_l2381_238119

/-- Given that z = (2 + mi) / (1 + i) is a pure imaginary number, 
    prove that the imaginary part of z is -2. -/
theorem imaginary_part_of_pure_imaginary_z (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → Complex.im z = -2 := by
  sorry

end imaginary_part_of_pure_imaginary_z_l2381_238119


namespace power_base_property_l2381_238123

theorem power_base_property (k : ℝ) (h : k > 1) :
  let x := k^(1/(k-1))
  ∀ y : ℝ, (k*x)^(y/k) = x^y := by
sorry

end power_base_property_l2381_238123


namespace price_after_nine_years_l2381_238124

/-- The price of a product after a certain number of three-year periods, given an initial price and a decay rate. -/
def price_after_periods (initial_price : ℝ) (decay_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decay_rate) ^ periods

/-- Theorem stating that if a product's price decreases by 25% every three years and its current price is 640 yuan, then its price after 9 years will be 270 yuan. -/
theorem price_after_nine_years :
  let initial_price : ℝ := 640
  let decay_rate : ℝ := 0.25
  let periods : ℕ := 3
  price_after_periods initial_price decay_rate periods = 270 := by
  sorry


end price_after_nine_years_l2381_238124


namespace sequence_determination_l2381_238154

/-- A sequence is determined if its terms are uniquely defined by given conditions -/
def is_determined (a : ℕ → ℝ) : Prop := sorry

/-- Arithmetic sequence with given S₁ and S₂ -/
def arithmetic_sequence (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ S₁ = a 1 ∧ S₂ = a 1 + a 2

/-- Geometric sequence with given S₁ and S₂ -/
def geometric_sequence_S₁S₂ (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₂ = a 1 + a 1 * q

/-- Geometric sequence with given S₁ and S₃ -/
def geometric_sequence_S₁S₃ (a : ℕ → ℝ) (S₁ S₃ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₃ = a 1 + a 1 * q + a 1 * q^2

/-- Sequence satisfying given recurrence relations -/
def recurrence_sequence (a : ℕ → ℝ) (x y c : ℝ) : Prop :=
  a 1 = c ∧ 
  (∀ n : ℕ, a (2*n + 2) = a (2*n) + x ∧ a (2*n + 1) = a (2*n - 1) + y)

theorem sequence_determination :
  ∀ a : ℕ → ℝ, ∀ S₁ S₂ S₃ x y c : ℝ,
  (is_determined a ↔ arithmetic_sequence a S₁ S₂) ∧
  (is_determined a ↔ geometric_sequence_S₁S₂ a S₁ S₂) ∧
  ¬(is_determined a ↔ geometric_sequence_S₁S₃ a S₁ S₃) ∧
  ¬(is_determined a ↔ recurrence_sequence a x y c) :=
sorry

end sequence_determination_l2381_238154


namespace third_student_weight_l2381_238169

theorem third_student_weight (original_count : ℕ) (original_avg : ℝ) 
  (new_count : ℕ) (new_avg : ℝ) (first_weight : ℝ) (second_weight : ℝ) :
  original_count = 29 →
  original_avg = 28 →
  new_count = original_count + 3 →
  new_avg = 27.3 →
  first_weight = 20 →
  second_weight = 30 →
  ∃ (third_weight : ℝ),
    third_weight = new_count * new_avg - original_count * original_avg - first_weight - second_weight ∧
    third_weight = 11.6 := by
  sorry

end third_student_weight_l2381_238169


namespace garden_area_difference_l2381_238142

-- Define the dimensions of the gardens
def karl_length : ℝ := 20
def karl_width : ℝ := 45
def makenna_length : ℝ := 25
def makenna_width : ℝ := 40

-- Define the areas of the gardens
def karl_area : ℝ := karl_length * karl_width
def makenna_area : ℝ := makenna_length * makenna_width

-- Theorem to prove
theorem garden_area_difference : makenna_area - karl_area = 100 := by
  sorry

end garden_area_difference_l2381_238142


namespace monday_rainfall_duration_l2381_238103

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_rate : ℝ
  monday_duration : ℝ
  tuesday_rate : ℝ
  tuesday_duration : ℝ
  wednesday_rate : ℝ
  wednesday_duration : ℝ
  total_rainfall : ℝ

/-- Theorem: The duration of rainfall on Monday is 7 hours -/
theorem monday_rainfall_duration (data : RainfallData) : data.monday_duration = 7 :=
  by
  have h1 : data.monday_rate = 1 := by sorry
  have h2 : data.tuesday_rate = 2 := by sorry
  have h3 : data.tuesday_duration = 4 := by sorry
  have h4 : data.wednesday_rate = 2 * data.tuesday_rate := by sorry
  have h5 : data.wednesday_duration = 2 := by sorry
  have h6 : data.total_rainfall = 23 := by sorry
  have h7 : data.total_rainfall = 
    data.monday_rate * data.monday_duration + 
    data.tuesday_rate * data.tuesday_duration + 
    data.wednesday_rate * data.wednesday_duration := by sorry
  sorry

end monday_rainfall_duration_l2381_238103


namespace boys_exam_pass_count_l2381_238111

theorem boys_exam_pass_count :
  ∀ (total_boys : ℕ) 
    (avg_all avg_pass avg_fail : ℚ)
    (pass_count : ℕ),
  total_boys = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  pass_count ≤ total_boys →
  (pass_count : ℚ) * avg_pass + (total_boys - pass_count : ℚ) * avg_fail = (total_boys : ℚ) * avg_all →
  pass_count = 100 := by
sorry

end boys_exam_pass_count_l2381_238111


namespace opposite_abs_neg_five_l2381_238172

theorem opposite_abs_neg_five : -(abs (-5)) = -5 := by sorry

end opposite_abs_neg_five_l2381_238172


namespace reggie_free_throws_l2381_238192

/-- Represents the number of points for each type of shot --/
structure PointValues where
  layup : ℕ
  freeThrow : ℕ
  longShot : ℕ

/-- Represents the shots made by a player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points scored by a player --/
def calculatePoints (pv : PointValues) (sm : ShotsMade) : ℕ :=
  pv.layup * sm.layups + pv.freeThrow * sm.freeThrows + pv.longShot * sm.longShots

theorem reggie_free_throws 
  (pointValues : PointValues)
  (reggieShotsMade : ShotsMade)
  (brotherShotsMade : ShotsMade)
  (h1 : pointValues.layup = 1)
  (h2 : pointValues.freeThrow = 2)
  (h3 : pointValues.longShot = 3)
  (h4 : reggieShotsMade.layups = 3)
  (h5 : reggieShotsMade.longShots = 1)
  (h6 : brotherShotsMade.layups = 0)
  (h7 : brotherShotsMade.freeThrows = 0)
  (h8 : brotherShotsMade.longShots = 4)
  (h9 : calculatePoints pointValues brotherShotsMade = calculatePoints pointValues reggieShotsMade + 2) :
  reggieShotsMade.freeThrows = 2 := by
  sorry

#check reggie_free_throws

end reggie_free_throws_l2381_238192


namespace ellipse_k_range_l2381_238186

theorem ellipse_k_range : 
  ∀ k : ℝ, (∃ x y : ℝ, (x^2 / (k - 2) + y^2 / (3 - k) = 1) ∧ 
  ((k - 2 > 0) ∧ (3 - k > 0) ∧ (k - 2 ≠ 3 - k))) → 
  (k > 2 ∧ k < 3 ∧ k ≠ 5/2) :=
by sorry

end ellipse_k_range_l2381_238186


namespace area_of_triangle_FYG_l2381_238144

theorem area_of_triangle_FYG (EF GH : ℝ) (area_EFGH : ℝ) (angle_E : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 400 →
  angle_E = 30 * π / 180 →
  ∃ (area_FYG : ℝ), area_FYG = 240 - 45 * Real.sqrt 3 :=
by sorry

end area_of_triangle_FYG_l2381_238144


namespace sum_P_2_neg_2_l2381_238159

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  P_0 : P 0 = k
  P_1 : P 1 = 3 * k
  P_neg_1 : P (-1) = 4 * k

/-- The sum of P(2) and P(-2) for a cubic polynomial with specific properties -/
theorem sum_P_2_neg_2 (k : ℝ) (P : CubicPolynomial k) :
  P.P 2 + P.P (-2) = 24 * k := by sorry

end sum_P_2_neg_2_l2381_238159


namespace trap_speed_constant_and_eight_l2381_238196

/-- Representation of a 4-level staircase --/
structure Staircase :=
  (h : ℝ)  -- height of each step
  (b : ℝ)  -- width of each step
  (a : ℝ)  -- length of the staircase
  (v : ℝ)  -- speed of the mouse

/-- The speed of the mouse trap required to catch the mouse --/
def trap_speed (s : Staircase) : ℝ := 8

/-- Theorem stating that the trap speed is constant and equal to 8 cm/s --/
theorem trap_speed_constant_and_eight (s : Staircase) 
  (h_height : s.h = 3)
  (h_width : s.b = 1)
  (h_length : s.a = 8)
  (h_mouse_speed : s.v = 17) :
  trap_speed s = 8 ∧ 
  ∀ (placement : ℝ), 0 ≤ placement ∧ placement ≤ s.a → trap_speed s = 8 := by
  sorry

#check trap_speed_constant_and_eight

end trap_speed_constant_and_eight_l2381_238196


namespace f_decreasing_interval_f_extremum_at_3_l2381_238176

/-- The function f(x) = 2x³ - 15x² + 36x - 24 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 15 * x^2 + 36 * x - 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 30 * x + 36

/-- Theorem stating that the decreasing interval of f is (2, 3) -/
theorem f_decreasing_interval :
  ∀ x : ℝ, (2 < x ∧ x < 3) ↔ (f' x < 0) :=
sorry

/-- Theorem stating that f has an extremum at x = 3 -/
theorem f_extremum_at_3 :
  f' 3 = 0 :=
sorry

end f_decreasing_interval_f_extremum_at_3_l2381_238176


namespace system_solutions_l2381_238156

theorem system_solutions :
  let S : Set (ℝ × ℝ × ℝ) := { (x, y, z) | x^5 = y^3 + 2*z ∧ y^5 = z^3 + 2*x ∧ z^5 = x^3 + 2*y }
  S = {(0, 0, 0), (Real.sqrt 2, Real.sqrt 2, Real.sqrt 2), (-Real.sqrt 2, -Real.sqrt 2, -Real.sqrt 2)} := by
  sorry

end system_solutions_l2381_238156


namespace max_value_of_m_l2381_238198

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0)
  (heq : 5 = m^2 * (a^2/b^2 + b^2/a^2) + m * (a/b + b/a)) :
  m ≤ (-1 + Real.sqrt 21) / 2 :=
by sorry

end max_value_of_m_l2381_238198


namespace troy_computer_purchase_l2381_238107

/-- The problem of Troy buying a new computer -/
theorem troy_computer_purchase (new_computer_cost initial_savings old_computer_value : ℕ)
  (h1 : new_computer_cost = 80)
  (h2 : initial_savings = 50)
  (h3 : old_computer_value = 20) :
  new_computer_cost - (initial_savings + old_computer_value) = 10 := by
  sorry

end troy_computer_purchase_l2381_238107


namespace equation_solution_l2381_238112

theorem equation_solution : ∃ x : ℝ, 15 * 2 = x - 3 + 5 ∧ x = 28 := by
  sorry

end equation_solution_l2381_238112


namespace phone_call_duration_l2381_238168

/-- Calculates the duration of a phone call given the initial credit, cost per minute, and remaining credit -/
theorem phone_call_duration (initial_credit remaining_credit cost_per_minute : ℚ) : 
  initial_credit = 30 ∧ 
  cost_per_minute = 16/100 ∧ 
  remaining_credit = 264/10 →
  (initial_credit - remaining_credit) / cost_per_minute = 22 := by
  sorry

end phone_call_duration_l2381_238168


namespace number_puzzle_l2381_238106

theorem number_puzzle : ∃! x : ℝ, x / 5 + 7 = x / 4 - 7 := by sorry

end number_puzzle_l2381_238106


namespace rational_roots_of_equation_l2381_238153

theorem rational_roots_of_equation (a b c d : ℝ) :
  ∃ x : ℚ, (a + b)^2 * (x + c^2) * (x + d^2) - (c + d)^2 * (x + a^2) * (x + b^2) = 0 :=
by sorry

end rational_roots_of_equation_l2381_238153


namespace ellipse_hyperbola_tangency_l2381_238116

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y n : ℝ) : Prop := x^2 - n*(y-1)^2 = 4

-- Define the tangency condition
def are_tangent (n : ℝ) : Prop := 
  ∃ x y, ellipse x y ∧ hyperbola x y n

-- Theorem statement
theorem ellipse_hyperbola_tangency (n : ℝ) :
  are_tangent n → n = 45/4 := by
  sorry

end ellipse_hyperbola_tangency_l2381_238116


namespace estimate_viewers_l2381_238134

theorem estimate_viewers (total_population : ℕ) (sample_size : ℕ) (sample_viewers : ℕ) 
  (h1 : total_population = 3600)
  (h2 : sample_size = 200)
  (h3 : sample_viewers = 160) :
  (total_population : ℚ) * (sample_viewers : ℚ) / (sample_size : ℚ) = 2880 := by
  sorry

end estimate_viewers_l2381_238134


namespace sqrt_five_irrational_and_greater_than_two_l2381_238194

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x ^ 2 = 5 := by
  sorry

end sqrt_five_irrational_and_greater_than_two_l2381_238194


namespace minimal_fraction_difference_l2381_238117

theorem minimal_fraction_difference (p q : ℕ+) : 
  (4 : ℚ) / 7 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 7 / 12 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 7 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < 7 / 12 → q ≤ q') →
  q - p = 8 := by
sorry

end minimal_fraction_difference_l2381_238117


namespace soccer_team_win_percentage_l2381_238182

/-- Calculate the percentage of games won by a soccer team -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 130) 
  (h2 : games_won = 78) : 
  (games_won : ℚ) / total_games * 100 = 60 := by
  sorry

end soccer_team_win_percentage_l2381_238182


namespace remaining_cooking_time_l2381_238128

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end remaining_cooking_time_l2381_238128


namespace even_function_max_value_l2381_238164

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_max_value
  (f : ℝ → ℝ)
  (h_even : IsEven f)
  (h_max : ∀ x ∈ Set.Icc (-2) (-1), f x ≤ -2)
  (h_attains : ∃ x ∈ Set.Icc (-2) (-1), f x = -2) :
  (∀ x ∈ Set.Icc 1 2, f x ≤ -2) ∧ (∃ x ∈ Set.Icc 1 2, f x = -2) :=
sorry

end even_function_max_value_l2381_238164


namespace ads_on_first_page_l2381_238177

theorem ads_on_first_page (page1 page2 page3 page4 : ℕ) : 
  page2 = 2 * page1 →
  page3 = page2 + 24 →
  page4 = 3 * page2 / 4 →
  68 = 2 * (page1 + page2 + page3 + page4) / 3 →
  page1 = 12 := by
sorry

end ads_on_first_page_l2381_238177


namespace work_completion_time_proof_l2381_238197

/-- Represents the time in days for a person to complete the work alone -/
structure WorkTime :=
  (days : ℚ)
  (days_pos : days > 0)

/-- Represents the combined work rate of multiple people -/
def combined_work_rate (work_times : List WorkTime) : ℚ :=
  work_times.map (λ wt => 1 / wt.days) |> List.sum

/-- The time required for the group to complete the work together -/
def group_work_time (work_times : List WorkTime) : ℚ :=
  1 / combined_work_rate work_times

theorem work_completion_time_proof 
  (david_time : WorkTime)
  (john_time : WorkTime)
  (mary_time : WorkTime)
  (h1 : david_time.days = 5)
  (h2 : john_time.days = 9)
  (h3 : mary_time.days = 7) :
  ⌈group_work_time [david_time, john_time, mary_time]⌉ = 3 := by
  sorry

#eval ⌈(315 : ℚ) / 143⌉

end work_completion_time_proof_l2381_238197


namespace sam_long_sleeve_shirts_l2381_238135

/-- Given information about Sam's shirts to wash -/
structure ShirtWashing where
  short_sleeve : ℕ
  washed : ℕ
  unwashed : ℕ

/-- The number of long sleeve shirts Sam had to wash -/
def long_sleeve_shirts (s : ShirtWashing) : ℕ :=
  s.washed + s.unwashed - s.short_sleeve

/-- Theorem stating the number of long sleeve shirts Sam had to wash -/
theorem sam_long_sleeve_shirts :
  ∀ s : ShirtWashing,
  s.short_sleeve = 40 →
  s.washed = 29 →
  s.unwashed = 34 →
  long_sleeve_shirts s = 23 := by
  sorry

end sam_long_sleeve_shirts_l2381_238135


namespace absolute_value_inequality_l2381_238145

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 := by
  sorry

end absolute_value_inequality_l2381_238145


namespace inequality_system_solution_iff_l2381_238126

theorem inequality_system_solution_iff (a : ℝ) :
  (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := by
  sorry

end inequality_system_solution_iff_l2381_238126


namespace inequality_proof_l2381_238118

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 0.5) :
  (x * y^2) / (x^3 + 1) + (y * z^2) / (y^3 + 1) + (z * x^2) / (z^3 + 1) ≥ 1 := by
  sorry

end inequality_proof_l2381_238118


namespace power_zero_of_three_minus_pi_l2381_238120

theorem power_zero_of_three_minus_pi : (3 - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end power_zero_of_three_minus_pi_l2381_238120


namespace smallest_tangent_circle_l2381_238130

/-- The line to which the circle is tangent -/
def line (x y : ℝ) : ℝ := x - y - 4

/-- The circle to which the target circle is tangent -/
def given_circle (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - 2*y

/-- The equation of the target circle -/
def target_circle (x y : ℝ) : ℝ := (x - 1)^2 + (y + 1)^2 - 2

/-- Theorem stating that the target circle is the smallest circle tangent to both the line and the given circle -/
theorem smallest_tangent_circle :
  ∀ r > 0, ∀ a b : ℝ,
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → line x y ≠ 0) ∧
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → given_circle x y ≠ 0) →
    r^2 ≥ 2 :=
sorry

end smallest_tangent_circle_l2381_238130


namespace caleb_spent_correct_amount_l2381_238143

-- Define the given conditions
def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 37

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  (double_burgers_bought * double_burger_cost) +
  ((total_burgers - double_burgers_bought) * single_burger_cost)

-- Theorem to prove
theorem caleb_spent_correct_amount :
  total_cost = 68.5 := by sorry

end caleb_spent_correct_amount_l2381_238143


namespace no_integer_solutions_l2381_238129

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 2*x*y + 9*z^2 = 150) := by
  sorry

end no_integer_solutions_l2381_238129


namespace inverse_matrices_solution_l2381_238140

/-- Given two 2x2 matrices that are inverses of each other, prove that a = 6 and b = 3/25 -/
theorem inverse_matrices_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1/5, -1/5; b, 2/5]
  A * B = 1 → a = 6 ∧ b = 3/25 := by
  sorry

#check inverse_matrices_solution

end inverse_matrices_solution_l2381_238140


namespace sugar_amount_is_two_l2381_238187

-- Define the ratios and quantities
def sugar_to_cheese_ratio : ℚ := 1 / 4
def vanilla_to_cheese_ratio : ℚ := 1 / 2
def eggs_to_vanilla_ratio : ℚ := 2
def eggs_used : ℕ := 8

-- Define the function to calculate sugar used
def sugar_used (eggs : ℕ) : ℚ :=
  (eggs : ℚ) / eggs_to_vanilla_ratio / vanilla_to_cheese_ratio * sugar_to_cheese_ratio

-- Theorem statement
theorem sugar_amount_is_two : sugar_used eggs_used = 2 := by
  sorry

end sugar_amount_is_two_l2381_238187


namespace right_triangle_sets_l2381_238162

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that the given sets of numbers satisfy or don't satisfy the right triangle condition
theorem right_triangle_sets :
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle (6/5) 2 (8/5)) ∧
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle (Real.sqrt 8) 2 (Real.sqrt 5)) :=
by sorry

end right_triangle_sets_l2381_238162


namespace bicycle_cost_price_l2381_238165

/-- Proves that the cost price of a bicycle for seller A is 150 given the selling conditions --/
theorem bicycle_cost_price
  (profit_A_to_B : ℝ) -- Profit percentage when A sells to B
  (profit_B_to_C : ℝ) -- Profit percentage when B sells to C
  (price_C : ℝ)       -- Price C pays for the bicycle
  (h1 : profit_A_to_B = 20)
  (h2 : profit_B_to_C = 25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B / 100) * (1 + profit_B_to_C / 100) :=
by sorry

end bicycle_cost_price_l2381_238165


namespace inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l2381_238104

-- Statement 1
theorem inequality_one (x : ℝ) (h : x ≥ 0) : x + 1 + 1 / (x + 1) ≥ 2 := by sorry

-- Statement 2
theorem inequality_two (x : ℝ) (h : x > 0) : (x + 1) / Real.sqrt x ≥ 2 := by sorry

-- Statement 3
theorem min_value_reciprocal : ∃ (m : ℝ), ∀ (x : ℝ), x + 1/x ≥ m ∧ ∃ (y : ℝ), y + 1/y = m := by sorry

-- Statement 4
theorem min_value_sqrt : ∃ (m : ℝ), ∀ (x : ℝ), Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ m ∧ 
  ∃ (y : ℝ), Real.sqrt (y^2 + 2) + 1 / Real.sqrt (y^2 + 2) = m := by sorry

end inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l2381_238104


namespace reflection_line_sum_l2381_238115

/-- 
Given a line y = mx + b, if the reflection of point (-4, 0) across this line 
is (2, 6), then m + b = 1.
-/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x = -1 ∧ y = 3) ↔ 
    (x = ((-4 + 2) / 2) ∧ y = ((0 + 6) / 2))) →
  (m = -1) →
  (m + b = 1) := by
  sorry

end reflection_line_sum_l2381_238115


namespace binomial_square_derivation_l2381_238152

theorem binomial_square_derivation (x y : ℝ) :
  ∃ (a b : ℝ), (-1/2 * x + y) * (y + 1/2 * x) = a^2 - b^2 :=
sorry

end binomial_square_derivation_l2381_238152


namespace inequality_proof_l2381_238181

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end inequality_proof_l2381_238181


namespace credibility_is_97_5_percent_l2381_238178

/-- Critical values table -/
def critical_values : List (Float × Float) := [
  (0.15, 2.072),
  (0.10, 2.706),
  (0.05, 3.841),
  (0.025, 5.024),
  (0.010, 6.635),
  (0.001, 10.828)
]

/-- The calculated K^2 value -/
def K_squared : Float := 6.109

/-- Function to determine credibility based on K^2 value and critical values table -/
def determine_credibility (K_sq : Float) (crit_vals : List (Float × Float)) : Float :=
  let lower_bound := crit_vals.find? (fun (p, k) => K_sq > k)
  let upper_bound := crit_vals.find? (fun (p, k) => K_sq ≤ k)
  match lower_bound, upper_bound with
  | some (p_lower, _), some (p_upper, _) => 100 * (1 - p_lower)
  | _, _ => 0  -- Default case if bounds are not found

/-- Theorem stating the credibility of the relationship -/
theorem credibility_is_97_5_percent :
  determine_credibility K_squared critical_values = 97.5 :=
sorry

end credibility_is_97_5_percent_l2381_238178


namespace least_n_satisfying_inequality_l2381_238191

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 := by
  sorry

end least_n_satisfying_inequality_l2381_238191


namespace volume_of_specific_room_l2381_238113

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m x 10 m x 10 m is 100,000 cubic meters -/
theorem volume_of_specific_room : 
  room_volume 100 10 10 = 100000 := by
  sorry

end volume_of_specific_room_l2381_238113


namespace combined_teaching_experience_l2381_238109

/-- Given two teachers, James and his partner, where James has taught for 40 years
    and his partner has taught for 10 years less than James, 
    their combined teaching experience is 70 years. -/
theorem combined_teaching_experience : 
  ∀ (james_experience partner_experience : ℕ),
  james_experience = 40 →
  partner_experience = james_experience - 10 →
  james_experience + partner_experience = 70 :=
by
  sorry

end combined_teaching_experience_l2381_238109


namespace M_subset_N_l2381_238146

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)}

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l2381_238146


namespace weight_of_b_l2381_238158

/-- Given three weights a, b, and c, prove that b equals 60 when:
    1. The average of a, b, and c is 60.
    2. The average of a and b is 70.
    3. The average of b and c is 50. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)
  (h2 : (a + b) / 2 = 70)
  (h3 : (b + c) / 2 = 50) : 
  b = 60 := by
  sorry

end weight_of_b_l2381_238158


namespace total_money_sam_and_billy_l2381_238100

/-- Given Sam has $75 and Billy has $25 less than twice the money Sam has, 
    their total money together is $200. -/
theorem total_money_sam_and_billy : 
  ∀ (sam_money billy_money : ℕ),
  sam_money = 75 →
  billy_money = 2 * sam_money - 25 →
  sam_money + billy_money = 200 := by
  sorry

end total_money_sam_and_billy_l2381_238100


namespace paper_fold_cut_ratio_l2381_238163

theorem paper_fold_cut_ratio : 
  let square_side : ℝ := 6
  let fold_ratio : ℝ := 1/3
  let cut_ratio : ℝ := 2/3
  let small_width : ℝ := square_side * fold_ratio
  let large_width : ℝ := square_side * (1 - fold_ratio) * (1 - cut_ratio)
  let small_perimeter : ℝ := 2 * (square_side + small_width)
  let large_perimeter : ℝ := 2 * (square_side + large_width)
  small_perimeter / large_perimeter = 12/17 := by
sorry

end paper_fold_cut_ratio_l2381_238163
