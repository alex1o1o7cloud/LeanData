import Mathlib

namespace NUMINAMATH_CALUDE_wall_length_proof_l970_97079

/-- Given a wall with specified dimensions and number of bricks, prove its length --/
theorem wall_length_proof (wall_height : ℝ) (wall_thickness : ℝ) 
  (brick_count : ℝ) (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) :
  wall_height = 100 →
  wall_thickness = 5 →
  brick_count = 242.42424242424244 →
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  (brick_length * brick_width * brick_height * brick_count) / (wall_height * wall_thickness) = 800 := by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l970_97079


namespace NUMINAMATH_CALUDE_f_monotonicity_and_tangent_intersection_l970_97038

/-- The function f(x) = x³ - x² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_tangent_intersection (a : ℝ) :
  (∀ x : ℝ, f' a x ≥ 0 → a ≥ 1/3) ∧
  (∃ t : ℝ, t * f' a 1 = f a 1 ∧ f a (-1) = -t * f' a (-1)) :=
sorry


end NUMINAMATH_CALUDE_f_monotonicity_and_tangent_intersection_l970_97038


namespace NUMINAMATH_CALUDE_box_long_side_length_l970_97065

/-- The length of the long sides of a box, given its dimensions and total velvet needed. -/
theorem box_long_side_length (total_velvet : ℝ) (short_side_length short_side_width : ℝ) 
  (long_side_width : ℝ) (top_bottom_area : ℝ) :
  total_velvet = 236 ∧
  short_side_length = 5 ∧
  short_side_width = 6 ∧
  long_side_width = 6 ∧
  top_bottom_area = 40 →
  ∃ long_side_length : ℝ,
    long_side_length = 8 ∧
    total_velvet = 2 * (short_side_length * short_side_width) + 
                   2 * top_bottom_area + 
                   2 * (long_side_length * long_side_width) :=
by sorry

end NUMINAMATH_CALUDE_box_long_side_length_l970_97065


namespace NUMINAMATH_CALUDE_perfect_square_iff_divisibility_l970_97085

theorem perfect_square_iff_divisibility (A : ℕ+) :
  (∃ d : ℕ+, A = d^2) ↔
  (∀ n : ℕ+, ∃ j : ℕ+, j ≤ n ∧ n ∣ ((A + j)^2 - A)) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_iff_divisibility_l970_97085


namespace NUMINAMATH_CALUDE_line_slope_l970_97094

theorem line_slope (x y : ℝ) : 4 * y + 2 * x = 10 → (y - 2.5) / x = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_slope_l970_97094


namespace NUMINAMATH_CALUDE_system_a_l970_97040

theorem system_a (x y : ℝ) : 
  y^4 + x*y^2 - 2*x^2 = 0 ∧ x + y = 6 →
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = -3) :=
sorry

end NUMINAMATH_CALUDE_system_a_l970_97040


namespace NUMINAMATH_CALUDE_nancy_shelving_problem_l970_97016

/-- The number of romance books shelved by Nancy the librarian --/
def romance_books : ℕ := 8

/-- The total number of books on the cart --/
def total_books : ℕ := 46

/-- The number of history books shelved --/
def history_books : ℕ := 12

/-- The number of poetry books shelved --/
def poetry_books : ℕ := 4

/-- The number of Western novels shelved --/
def western_books : ℕ := 5

/-- The number of biographies shelved --/
def biography_books : ℕ := 6

theorem nancy_shelving_problem :
  romance_books = 8 ∧
  total_books = 46 ∧
  history_books = 12 ∧
  poetry_books = 4 ∧
  western_books = 5 ∧
  biography_books = 6 ∧
  (total_books - (history_books + romance_books + poetry_books)) % 2 = 0 ∧
  (total_books - (history_books + romance_books + poetry_books)) / 2 = western_books + biography_books :=
by sorry

end NUMINAMATH_CALUDE_nancy_shelving_problem_l970_97016


namespace NUMINAMATH_CALUDE_expected_winnings_l970_97046

/-- Represents the outcome of rolling the die -/
inductive DieOutcome
  | Six
  | Odd
  | Even

/-- The probability of rolling a 6 -/
def prob_six : ℚ := 1/4

/-- The probability of rolling an odd number (1, 3, or 5) -/
def prob_odd : ℚ := (1 - prob_six) * (3/5)

/-- The probability of rolling an even number (2 or 4) -/
def prob_even : ℚ := (1 - prob_six) * (2/5)

/-- The payoff for each outcome -/
def payoff (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Six => -2
  | DieOutcome.Odd => 2
  | DieOutcome.Even => 4

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  prob_six * payoff DieOutcome.Six +
  prob_odd * payoff DieOutcome.Odd +
  prob_even * payoff DieOutcome.Even

theorem expected_winnings :
  expected_value = 8/5 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_l970_97046


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l970_97064

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (Real.pi + α) = 2/3) : 
  Real.cos (2 * α) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l970_97064


namespace NUMINAMATH_CALUDE_equation_roots_l970_97096

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (18 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l970_97096


namespace NUMINAMATH_CALUDE_sum_of_roots_l970_97042

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l970_97042


namespace NUMINAMATH_CALUDE_fraction_problem_l970_97077

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N * f^2 = 6^3 ∧ N * f^2 = 7776 → f = 1/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l970_97077


namespace NUMINAMATH_CALUDE_milk_savings_l970_97039

-- Define the problem parameters
def gallons : ℕ := 8
def original_price : ℚ := 3.20
def discount_rate : ℚ := 0.25

-- Define the function to calculate savings
def calculate_savings (g : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  g * p * d

-- Theorem statement
theorem milk_savings :
  calculate_savings gallons original_price discount_rate = 6.40 := by
  sorry


end NUMINAMATH_CALUDE_milk_savings_l970_97039


namespace NUMINAMATH_CALUDE_smallest_valid_circular_arrangement_l970_97033

/-- A function that checks if two natural numbers share at least one digit in their decimal representation -/
def shareDigit (a b : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def validArrangement (lst : List ℕ) : Prop := sorry

/-- The smallest natural number N ≥ 2 for which a valid circular arrangement exists -/
def smallestValidN : ℕ := 29

theorem smallest_valid_circular_arrangement :
  (smallestValidN ≥ 2) ∧
  (∃ (lst : List ℕ), lst.length = smallestValidN ∧ 
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ smallestValidN) ∧
    validArrangement lst) ∧
  (∀ N < smallestValidN, ¬∃ (lst : List ℕ), lst.length = N ∧
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ N) ∧
    validArrangement lst) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_circular_arrangement_l970_97033


namespace NUMINAMATH_CALUDE_inequality_system_solution_l970_97014

theorem inequality_system_solution (x : ℝ) :
  (3 * (x + 2) - x > 4 ∧ (1 + 2*x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l970_97014


namespace NUMINAMATH_CALUDE_b_range_l970_97007

theorem b_range (b : ℝ) : (∀ x : ℝ, x^2 + b*x + b > 0) → b ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l970_97007


namespace NUMINAMATH_CALUDE_triangle_properties_l970_97073

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 →
  (∃ (R : ℝ), 2 * R = c / Real.sin C ∧ 2 * R = b / Real.sin B ∧ 2 * R = a / Real.sin A) →
  (a = 2 ∧ b = 2) ∧
  (∀ (a' b' : ℝ), b' / 2 + a' ≤ 2 * Real.sqrt 21 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l970_97073


namespace NUMINAMATH_CALUDE_converse_ptolemy_l970_97009

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The length of a line segment between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Whether a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Ptolemy's condition for a quadrilateral -/
def ptolemy_condition (q : Quadrilateral) : Prop :=
  let AC := distance q.A q.C
  let BD := distance q.B q.D
  let AB := distance q.A q.B
  let CD := distance q.C q.D
  let AD := distance q.A q.D
  let BC := distance q.B q.C
  AC * BD = AB * CD + AD * BC

theorem converse_ptolemy (q : Quadrilateral) :
  ptolemy_condition q → is_cyclic q := by sorry

end NUMINAMATH_CALUDE_converse_ptolemy_l970_97009


namespace NUMINAMATH_CALUDE_chip_notebook_packs_l970_97008

/-- The number of packs of notebook paper Chip will use after 6 weeks -/
def notebook_packs_used (pages_per_class_per_day : ℕ) (num_classes : ℕ) 
  (days_per_week : ℕ) (sheets_per_pack : ℕ) (num_weeks : ℕ) : ℕ :=
  (pages_per_class_per_day * num_classes * days_per_week * num_weeks) / sheets_per_pack

/-- Theorem stating the number of packs Chip will use -/
theorem chip_notebook_packs : 
  notebook_packs_used 2 5 5 100 6 = 3 := by sorry

end NUMINAMATH_CALUDE_chip_notebook_packs_l970_97008


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l970_97004

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - 10

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = -10) ∧ (x₀ = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l970_97004


namespace NUMINAMATH_CALUDE_composition_properties_l970_97057

variable {X Y V : Type*}
variable (f : X → Y) (g : Y → V)

theorem composition_properties :
  ((∀ x₁ x₂ : X, g (f x₁) = g (f x₂) → x₁ = x₂) → (∀ x₁ x₂ : X, f x₁ = f x₂ → x₁ = x₂)) ∧
  ((∀ v : V, ∃ x : X, g (f x) = v) → (∀ v : V, ∃ y : Y, g y = v)) := by
  sorry

end NUMINAMATH_CALUDE_composition_properties_l970_97057


namespace NUMINAMATH_CALUDE_trajectory_difference_latitude_l970_97055

/-- The latitude at which the difference in trajectory lengths equals the height difference -/
theorem trajectory_difference_latitude (R h : ℝ) (θ : ℝ) 
  (h_pos : h > 0) 
  (r₁_def : R * Real.cos θ = R * Real.cos θ)
  (r₂_def : (R + h) * Real.cos θ = (R + h) * Real.cos θ)
  (s_def : 2 * Real.pi * (R + h) * Real.cos θ - 2 * Real.pi * R * Real.cos θ = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_difference_latitude_l970_97055


namespace NUMINAMATH_CALUDE_map_width_l970_97044

/-- The width of a rectangular map given its length and area -/
theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) :
  area / length = 10 := by
  sorry

end NUMINAMATH_CALUDE_map_width_l970_97044


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l970_97097

-- Define the ratio of Ram's weight to Shyam's weight
def weight_ratio : ℚ := 2 / 5

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 10 / 100

-- Define the total new weight
def total_new_weight : ℚ := 828 / 10

-- Define the total weight increase percentage
def total_increase : ℚ := 15 / 100

-- Function to calculate Shyam's weight increase percentage
def shyam_increase_percentage : ℚ := sorry

-- Theorem statement
theorem shyam_weight_increase :
  abs (shyam_increase_percentage - 1709 / 10000) < 1 / 1000 := by sorry

end NUMINAMATH_CALUDE_shyam_weight_increase_l970_97097


namespace NUMINAMATH_CALUDE_apple_juice_fraction_l970_97054

theorem apple_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_fullness pitcher2_fullness : ℚ) :
  pitcher1_capacity = 800 →
  pitcher2_capacity = 500 →
  pitcher1_fullness = 1/4 →
  pitcher2_fullness = 3/8 →
  (pitcher1_capacity * pitcher1_fullness + pitcher2_capacity * pitcher2_fullness) /
  (pitcher1_capacity + pitcher2_capacity) = 31/104 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_fraction_l970_97054


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l970_97061

/-- The number of combinations of k items chosen from a set of n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of available pizza toppings -/
def n : ℕ := 7

/-- The number of toppings to be chosen -/
def k : ℕ := 3

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35 -/
theorem pizza_toppings_combinations : binomial n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l970_97061


namespace NUMINAMATH_CALUDE_expression_evaluation_l970_97002

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x + x * y = 801 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l970_97002


namespace NUMINAMATH_CALUDE_smallest_reciprocal_l970_97013

theorem smallest_reciprocal (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_order : a > b ∧ b > c) :
  (1 : ℚ) / a < (1 : ℚ) / b ∧ (1 : ℚ) / b < (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_l970_97013


namespace NUMINAMATH_CALUDE_function_is_zero_l970_97032

/-- A function satisfying the given functional equation and non-negativity condition -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) ∧
  (∀ u : ℝ, f u ≥ 0)

/-- The main theorem stating that any function satisfying the conditions must be identically zero -/
theorem function_is_zero (f : ℝ → ℝ) (hf : SatisfiesFunctionalEquation f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l970_97032


namespace NUMINAMATH_CALUDE_emily_purchase_cost_l970_97088

/-- Calculate the total cost of Emily's purchase including discount, tax, and installation fee -/
theorem emily_purchase_cost :
  let curtain_price : ℚ := 30
  let curtain_quantity : ℕ := 2
  let print_price : ℚ := 15
  let print_quantity : ℕ := 9
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.08
  let installation_fee : ℚ := 50

  let subtotal : ℚ := curtain_price * curtain_quantity + print_price * print_quantity
  let discounted_total : ℚ := subtotal * (1 - discount_rate)
  let taxed_total : ℚ := discounted_total * (1 + tax_rate)
  let total_cost : ℚ := taxed_total + installation_fee

  total_cost = 239.54 := by sorry

end NUMINAMATH_CALUDE_emily_purchase_cost_l970_97088


namespace NUMINAMATH_CALUDE_function_set_property_l970_97069

/-- A set of functions from ℝ to ℝ satisfying a specific property -/
def FunctionSet : Type := {A : Set (ℝ → ℝ) // 
  ∀ (f₁ f₂ : ℝ → ℝ), f₁ ∈ A → f₂ ∈ A → 
    ∃ (f₃ : ℝ → ℝ), f₃ ∈ A ∧ 
      ∀ (x y : ℝ), f₁ (f₂ y - x) + 2 * x = f₃ (x + y)}

/-- The main theorem -/
theorem function_set_property (A : FunctionSet) :
  ∀ (f : ℝ → ℝ), f ∈ A.val → ∀ (x : ℝ), f (x - f x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_set_property_l970_97069


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_4x_l970_97099

theorem factorization_x_squared_minus_4x (x : ℝ) : x^2 - 4*x = x*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_4x_l970_97099


namespace NUMINAMATH_CALUDE_delivery_time_problem_l970_97028

/-- Calculates the time needed to deliver all cars -/
def delivery_time (coal_cars iron_cars wood_cars : ℕ) 
                  (coal_deposit iron_deposit wood_deposit : ℕ) 
                  (time_between_stations : ℕ) : ℕ :=
  let coal_stations := (coal_cars + coal_deposit - 1) / coal_deposit
  let iron_stations := (iron_cars + iron_deposit - 1) / iron_deposit
  let wood_stations := (wood_cars + wood_deposit - 1) / wood_deposit
  let max_stations := max coal_stations (max iron_stations wood_stations)
  max_stations * time_between_stations

/-- Proves that the delivery time for the given problem is 100 minutes -/
theorem delivery_time_problem : 
  delivery_time 6 12 2 2 3 1 25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_delivery_time_problem_l970_97028


namespace NUMINAMATH_CALUDE_student_arrangement_count_l970_97090

/-- The number of ways to arrange 6 students with specific constraints -/
def arrangement_count : ℕ := 144

/-- Two specific students (A and B) must be adjacent -/
def adjacent_pair : ℕ := 2

/-- Number of students excluding A, B, and C -/
def other_students : ℕ := 3

/-- Number of valid positions for student C -/
def valid_positions_for_c : ℕ := 3

theorem student_arrangement_count :
  arrangement_count = 
    (Nat.factorial other_students) * 
    (Nat.factorial (other_students + 1) / Nat.factorial (other_students - 1)) * 
    adjacent_pair := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l970_97090


namespace NUMINAMATH_CALUDE_equation_true_iff_m_zero_l970_97074

theorem equation_true_iff_m_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_true_iff_m_zero_l970_97074


namespace NUMINAMATH_CALUDE_power_multiplication_l970_97059

theorem power_multiplication (x : ℝ) : x^3 * x^4 = x^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l970_97059


namespace NUMINAMATH_CALUDE_euclids_lemma_l970_97045

theorem euclids_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) : p ∣ a ∨ p ∣ b := by
  sorry

-- Gauss's lemma (given)
axiom gauss_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) (hna : ¬(p ∣ a)) : p ∣ b

end NUMINAMATH_CALUDE_euclids_lemma_l970_97045


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l970_97006

/-- Calculates the total revenue of a movie theater given ticket sales and pricing information. -/
def calculate_total_revenue (
  matinee_price : ℚ)
  (evening_price : ℚ)
  (threeD_price : ℚ)
  (evening_group_discount : ℚ)
  (threeD_online_surcharge : ℚ)
  (early_bird_discount : ℚ)
  (matinee_tickets : ℕ)
  (early_bird_tickets : ℕ)
  (evening_tickets : ℕ)
  (evening_group_tickets : ℕ)
  (threeD_tickets : ℕ)
  (threeD_online_tickets : ℕ) : ℚ :=
  sorry

theorem movie_theater_revenue :
  let matinee_price : ℚ := 5
  let evening_price : ℚ := 12
  let threeD_price : ℚ := 20
  let evening_group_discount : ℚ := 0.1
  let threeD_online_surcharge : ℚ := 2
  let early_bird_discount : ℚ := 0.5
  let matinee_tickets : ℕ := 200
  let early_bird_tickets : ℕ := 20
  let evening_tickets : ℕ := 300
  let evening_group_tickets : ℕ := 150
  let threeD_tickets : ℕ := 100
  let threeD_online_tickets : ℕ := 60
  calculate_total_revenue
    matinee_price evening_price threeD_price
    evening_group_discount threeD_online_surcharge early_bird_discount
    matinee_tickets early_bird_tickets evening_tickets
    evening_group_tickets threeD_tickets threeD_online_tickets = 6490 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_l970_97006


namespace NUMINAMATH_CALUDE_eugene_pencils_l970_97005

theorem eugene_pencils (initial_pencils : ℝ) (pencils_given : ℝ) :
  initial_pencils = 51.0 →
  pencils_given = 6.0 →
  initial_pencils - pencils_given = 45.0 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l970_97005


namespace NUMINAMATH_CALUDE_unique_k_value_l970_97029

/-- A predicate to check if a number is a non-zero digit -/
def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The expression as a function of k and t -/
def expression (k t : ℕ) : ℤ := 8 * k * 100 + 8 + k * 100 + 88 - 16 * t * 10 - 6

theorem unique_k_value :
  ∀ k t : ℕ,
  is_nonzero_digit k →
  is_nonzero_digit t →
  t = 6 →
  (∃ m : ℤ, expression k t = m) →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_unique_k_value_l970_97029


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l970_97037

theorem min_value_product_quotient (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k ≥ 2) :
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x*y*z) ≥ (2+k)^3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l970_97037


namespace NUMINAMATH_CALUDE_fish_eaten_l970_97043

theorem fish_eaten (initial_fish : ℕ) (temp_added : ℕ) (exchanged : ℕ) (final_fish : ℕ)
  (h1 : initial_fish = 14)
  (h2 : temp_added = 2)
  (h3 : exchanged = 3)
  (h4 : final_fish = 11) :
  initial_fish - (final_fish - exchanged) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fish_eaten_l970_97043


namespace NUMINAMATH_CALUDE_geometric_series_equation_solution_l970_97056

theorem geometric_series_equation_solution (x : ℝ) : 
  (|x| < 0.5) →
  (∑' n, (2*x)^n = 3.4 - 1.2*x) →
  x = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equation_solution_l970_97056


namespace NUMINAMATH_CALUDE_bell_peppers_needed_l970_97026

/-- Represents the number of slices and pieces obtained from one bell pepper -/
def slices_per_pepper : ℕ := 20

/-- Represents the fraction of large slices that are cut into smaller pieces -/
def fraction_cut : ℚ := 1/2

/-- Represents the number of smaller pieces each large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- Represents the total number of slices and pieces Tamia wants to use -/
def total_slices : ℕ := 200

/-- Proves that 5 bell peppers are needed to produce 200 slices and pieces -/
theorem bell_peppers_needed : 
  (total_slices : ℚ) / ((1 - fraction_cut) * slices_per_pepper + 
  fraction_cut * slices_per_pepper * pieces_per_slice) = 5 := by
sorry

end NUMINAMATH_CALUDE_bell_peppers_needed_l970_97026


namespace NUMINAMATH_CALUDE_total_amount_theorem_l970_97052

/-- Represents the types of books in the collection -/
inductive BookType
  | Novel
  | Biography
  | ScienceBook

/-- Calculates the total amount received from book sales -/
def calculateTotalAmount (totalBooks : ℕ) 
                         (soldPercentages : BookType → ℚ)
                         (prices : BookType → ℕ)
                         (remainingBooks : BookType → ℕ) : ℕ :=
  sorry

/-- The main theorem stating the total amount received from book sales -/
theorem total_amount_theorem (totalBooks : ℕ)
                             (soldPercentages : BookType → ℚ)
                             (prices : BookType → ℕ)
                             (remainingBooks : BookType → ℕ) : 
  totalBooks = 300 ∧
  soldPercentages BookType.Novel = 3/5 ∧
  soldPercentages BookType.Biography = 2/3 ∧
  soldPercentages BookType.ScienceBook = 7/10 ∧
  prices BookType.Novel = 4 ∧
  prices BookType.Biography = 7 ∧
  prices BookType.ScienceBook = 6 ∧
  remainingBooks BookType.Novel = 30 ∧
  remainingBooks BookType.Biography = 35 ∧
  remainingBooks BookType.ScienceBook = 25 →
  calculateTotalAmount totalBooks soldPercentages prices remainingBooks = 1018 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l970_97052


namespace NUMINAMATH_CALUDE_problem_1_l970_97041

theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_l970_97041


namespace NUMINAMATH_CALUDE_unique_g_30_equals_48_l970_97034

def sumOfDivisors (n : ℕ) : ℕ := sorry

def g₁ (n : ℕ) : ℕ := 4 * sumOfDivisors n

def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

theorem unique_g_30_equals_48 :
  ∃! n : ℕ, n ≤ 30 ∧ g 30 n = 48 := by sorry

end NUMINAMATH_CALUDE_unique_g_30_equals_48_l970_97034


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l970_97067

/-- Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers :
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l970_97067


namespace NUMINAMATH_CALUDE_sequence_not_periodic_l970_97051

/-- The sequence (a_n) defined by a_n = ⌊x^(n+1)⌋ - x⌊x^n⌋ is not periodic for any real x > 1 that is not an integer. -/
theorem sequence_not_periodic (x : ℝ) (hx : x > 1) (hx_not_int : ¬ ∃ n : ℤ, x = n) :
  ¬ ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, 
    (⌊x^(n+1)⌋ - x * ⌊x^n⌋ : ℝ) = (⌊x^(n+p+1)⌋ - x * ⌊x^(n+p)⌋ : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_periodic_l970_97051


namespace NUMINAMATH_CALUDE_tylers_puppies_l970_97035

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (h1 : num_dogs = 25) (h2 : puppies_per_dog = 8) :
  num_dogs * puppies_per_dog = 200 := by
  sorry

end NUMINAMATH_CALUDE_tylers_puppies_l970_97035


namespace NUMINAMATH_CALUDE_erased_number_theorem_l970_97087

theorem erased_number_theorem (n : ℕ) (h1 : n = 20) :
  ∀ x ∈ Finset.range n,
    (∃ y ∈ Finset.range n \ {x}, (n * (n + 1) / 2 - x : ℚ) / (n - 1) = y) ↔ 
    x = 1 ∨ x = n :=
by sorry

end NUMINAMATH_CALUDE_erased_number_theorem_l970_97087


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l970_97017

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l970_97017


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l970_97093

/-- Given a triangle ABC with AB = 16 and AC = 5, where the angle bisectors of ∠ABC and ∠BCA 
    meet at point P inside the triangle such that AP = 4, prove that BC = 14. -/
theorem triangle_angle_bisector_theorem (A B C P : ℝ × ℝ) : 
  let d (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- AB = 16
  d A B = 16 →
  -- AC = 5
  d A C = 5 →
  -- P is on the angle bisector of ∠ABC
  (d A P / d B P = d A C / d B C) →
  -- P is on the angle bisector of ∠BCA
  (d C P / d A P = d C B / d A B) →
  -- P is inside the triangle
  (0 < d A P ∧ d A P < d A B ∧ d A P < d A C) →
  -- AP = 4
  d A P = 4 →
  -- BC = 14
  d B C = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l970_97093


namespace NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l970_97063

/-- Represents the probability of Alex being paired with Jamie in a class pairing scenario -/
theorem alex_jamie_pairing_probability 
  (total_students : ℕ) 
  (paired_students : ℕ) 
  (h1 : total_students = 50) 
  (h2 : paired_students = 20) 
  (h3 : paired_students < total_students) :
  (1 : ℚ) / (total_students - paired_students - 1 : ℚ) = 1/29 := by
sorry

end NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l970_97063


namespace NUMINAMATH_CALUDE_inequality_solution_set_function_domain_set_l970_97083

-- Part 1: Inequality solution
def inequality_solution (x : ℝ) : Prop :=
  x * (x + 2) > x * (3 - x) + 6

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_solution x ↔ (x < -3/2 ∨ x > 2) :=
sorry

-- Part 2: Function domain
def function_domain (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 1 ∧ -x^2 - x + 6 > 0

theorem function_domain_set :
  ∀ x : ℝ, function_domain x ↔ (-1 ≤ x ∧ x < 2 ∧ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_function_domain_set_l970_97083


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_locus_l970_97019

/-- Given a hyperbola x^2 - y^2/4 = 1, if two perpendicular lines through its center O
    intersect the hyperbola at points A and B, then the locus of the midpoint P of chord AB
    satisfies the equation 3(4x^2 - y^2)^2 = 4(16x^2 + y^2). -/
theorem hyperbola_midpoint_locus (x y : ℝ) :
  (∃ (m n : ℝ),
    -- A and B are on the hyperbola
    4 * (x - m)^2 - (y - n)^2 = 4 ∧
    4 * (x + m)^2 - (y + n)^2 = 4 ∧
    -- OA ⊥ OB
    x^2 + y^2 = m^2 + n^2 ∧
    -- (x, y) is the midpoint of AB
    (x - m, y - n) = (-x - m, -y - n)) →
  3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) := by
sorry


end NUMINAMATH_CALUDE_hyperbola_midpoint_locus_l970_97019


namespace NUMINAMATH_CALUDE_wine_problem_equations_l970_97075

/-- Represents the number of guests intoxicated by one bottle of good wine -/
def good_wine_intoxication : ℚ := 3

/-- Represents the number of bottles of weak wine needed to intoxicate one guest -/
def weak_wine_intoxication : ℚ := 3

/-- Represents the total number of intoxicated guests -/
def total_intoxicated_guests : ℚ := 33

/-- Represents the total number of bottles of wine consumed -/
def total_bottles : ℚ := 19

/-- Represents the number of bottles of good wine -/
def x : ℚ := sorry

/-- Represents the number of bottles of weak wine -/
def y : ℚ := sorry

theorem wine_problem_equations :
  (x + y = total_bottles) ∧
  (good_wine_intoxication * x + (1 / weak_wine_intoxication) * y = total_intoxicated_guests) :=
by sorry

end NUMINAMATH_CALUDE_wine_problem_equations_l970_97075


namespace NUMINAMATH_CALUDE_max_value_ab_l970_97024

theorem max_value_ab (a b : ℝ) (g : ℝ → ℝ) (ha : a > 0) (hb : b > 0)
  (hg : ∀ x, g x = 2^x) (h_prod : g a * g b = 2) :
  ∀ x y, x > 0 → y > 0 → g x * g y = 2 → x * y ≤ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_value_ab_l970_97024


namespace NUMINAMATH_CALUDE_cone_volume_l970_97053

/-- Given a cone circumscribed by a sphere, proves that the volume of the cone is 3π -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  (π * r * l = 2 * π * r^2) →  -- lateral area is twice base area
  (4 * π * (2^2) = 16 * π) →   -- surface area of circumscribing sphere is 16π
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l970_97053


namespace NUMINAMATH_CALUDE_square_arrangement_sum_l970_97047

/-- The sum of integers from -12 to 18 inclusive -/
def total_sum : ℤ := 93

/-- The size of the square matrix -/
def matrix_size : ℕ := 6

/-- The common sum for each row, column, and main diagonal -/
def common_sum : ℚ := 15.5

theorem square_arrangement_sum :
  total_sum = matrix_size * (common_sum : ℚ).num / (common_sum : ℚ).den :=
sorry

end NUMINAMATH_CALUDE_square_arrangement_sum_l970_97047


namespace NUMINAMATH_CALUDE_concert_revenue_l970_97018

theorem concert_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_tickets : total_tickets = 200)
  (h_revenue : total_revenue = 3000) : ℕ :=
by
  -- Let f be the number of full-price tickets
  -- Let d be the number of discount tickets
  -- Let p be the price of a full-price ticket
  have h1 : ∃ (f d p : ℕ), 
    f + d = total_tickets ∧ 
    f * p + d * (p / 3) = total_revenue ∧ 
    f * p = 1500
  sorry

  exact 1500


end NUMINAMATH_CALUDE_concert_revenue_l970_97018


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l970_97092

/-- A geometric sequence with sum of first n terms S_n = 3 * 2^n + m has common ratio 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 3 * 2^n + (S 0 - 3)) : 
  ∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l970_97092


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l970_97084

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for part I
theorem solution_set_f (x : ℝ) : f x < 8 ↔ -5/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part II
theorem range_of_m (m : ℝ) : (∃ x, f x ≤ |3*m + 1|) → (m ≤ -5/3 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l970_97084


namespace NUMINAMATH_CALUDE_letters_theorem_l970_97066

def total_letters (brother_letters : ℕ) : ℕ :=
  let greta_letters := brother_letters + 10
  let mother_letters := 2 * (brother_letters + greta_letters)
  brother_letters + greta_letters + mother_letters

theorem letters_theorem : total_letters 40 = 270 := by
  sorry

end NUMINAMATH_CALUDE_letters_theorem_l970_97066


namespace NUMINAMATH_CALUDE_books_left_over_l970_97036

/-- Given a repacking scenario, proves the number of books left over -/
theorem books_left_over 
  (initial_boxes : ℕ) 
  (books_per_initial_box : ℕ) 
  (book_weight : ℕ) 
  (books_per_new_box : ℕ) 
  (max_new_box_weight : ℕ) 
  (h1 : initial_boxes = 1430)
  (h2 : books_per_initial_box = 42)
  (h3 : book_weight = 200)
  (h4 : books_per_new_box = 45)
  (h5 : max_new_box_weight = 9000)
  (h6 : books_per_new_box * book_weight ≤ max_new_box_weight) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

#check books_left_over

end NUMINAMATH_CALUDE_books_left_over_l970_97036


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l970_97068

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  let z := Complex.abs (3 + 4 * i) / (1 - 2 * i)
  Complex.im z = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l970_97068


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l970_97091

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : (R^2) / (r^2) = 4) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l970_97091


namespace NUMINAMATH_CALUDE_complement_of_M_l970_97021

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l970_97021


namespace NUMINAMATH_CALUDE_min_sum_dimensions_2310_l970_97086

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ :=
  d.length.val * d.width.val * d.height.val

/-- Calculates the sum of dimensions of a box -/
def sumDimensions (d : BoxDimensions) : ℕ :=
  d.length.val + d.width.val + d.height.val

/-- Theorem: The minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_sum_dimensions_2310 :
  (∃ d : BoxDimensions, volume d = 2310) →
  (∀ d : BoxDimensions, volume d = 2310 → sumDimensions d ≥ 42) ∧
  (∃ d : BoxDimensions, volume d = 2310 ∧ sumDimensions d = 42) :=
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_2310_l970_97086


namespace NUMINAMATH_CALUDE_half_obtuse_angle_in_first_quadrant_l970_97011

theorem half_obtuse_angle_in_first_quadrant (α : Real) (h : π / 2 < α ∧ α < π) :
  π / 4 < α / 2 ∧ α / 2 < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_obtuse_angle_in_first_quadrant_l970_97011


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l970_97049

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hnz : x > 0 ∨ y > 0 ∨ z > 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧ 
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ 
   (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t ∧ z = t) ∨ 
   (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = t ∧ y = t ∧ z = 1 - t)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l970_97049


namespace NUMINAMATH_CALUDE_inequality_proof_l970_97058

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 6) : 
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z))) + 
  (1 / (2 * Real.sqrt 2 + y^2 * (x + z))) + 
  (1 / (2 * Real.sqrt 2 + z^2 * (x + y))) ≤ 
  1 / (x * y * z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l970_97058


namespace NUMINAMATH_CALUDE_apollonius_circle_symmetric_x_axis_l970_97048

/-- Apollonius Circle -/
def ApolloniusCircle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x + 1)^2 + y^2 = a^2 * ((x - 1)^2 + y^2)}

/-- Symmetry about x-axis -/
def SymmetricAboutXAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (x, -y) ∈ S

theorem apollonius_circle_symmetric_x_axis (a : ℝ) (ha : a > 1) :
  SymmetricAboutXAxis (ApolloniusCircle a) := by
  sorry

end NUMINAMATH_CALUDE_apollonius_circle_symmetric_x_axis_l970_97048


namespace NUMINAMATH_CALUDE_green_peppers_weight_equal_pepper_weights_l970_97023

/-- The weight of green peppers bought by Dale's Vegetarian Restaurant -/
def green_peppers : ℝ := 2.8333333335

/-- The total weight of peppers bought by Dale's Vegetarian Restaurant -/
def total_peppers : ℝ := 5.666666667

/-- Theorem stating that the weight of green peppers is half the total weight of peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers / 2 :=
by sorry

/-- Theorem stating that the weight of green peppers is equal to the weight of red peppers -/
theorem equal_pepper_weights :
  green_peppers = total_peppers - green_peppers :=
by sorry

end NUMINAMATH_CALUDE_green_peppers_weight_equal_pepper_weights_l970_97023


namespace NUMINAMATH_CALUDE_min_value_theorem_l970_97081

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = 1 → 2/a + 5/b ≥ 2/x + 5/y) ∧ 2/x + 5/y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l970_97081


namespace NUMINAMATH_CALUDE_double_series_convergence_l970_97095

/-- The double series ∑_{m=1}^∞ ∑_{n=1}^∞ 1/(mn(m+n+2)) converges to 3/2. -/
theorem double_series_convergence :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_double_series_convergence_l970_97095


namespace NUMINAMATH_CALUDE_slope_from_angle_l970_97080

theorem slope_from_angle (θ : Real) (h : θ = 5 * Real.pi / 6) :
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_from_angle_l970_97080


namespace NUMINAMATH_CALUDE_flour_for_hundred_cookies_l970_97027

-- Define the recipe's ratio
def recipe_cookies : ℕ := 40
def recipe_flour : ℚ := 3

-- Define the desired number of cookies
def desired_cookies : ℕ := 100

-- Define the function to calculate required flour
def required_flour (cookies : ℕ) : ℚ :=
  (recipe_flour / recipe_cookies) * cookies

-- Theorem statement
theorem flour_for_hundred_cookies :
  required_flour desired_cookies = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_hundred_cookies_l970_97027


namespace NUMINAMATH_CALUDE_train_distance_problem_l970_97060

/-- Theorem: Train Distance Problem
Given:
- A passenger train travels from A to B at 60 km/h for 2/3 of the journey, then at 30 km/h for the rest.
- A high-speed train travels at 120 km/h and catches up with the passenger train 80 km before B.
Prove that the distance from A to B is 360 km. -/
theorem train_distance_problem (D : ℝ) 
  (h1 : D > 0)  -- Distance is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ (2/3 * D) / 60 + (1/3 * D) / 30 = (D - 80) / 120 + t)
  : D = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l970_97060


namespace NUMINAMATH_CALUDE_production_line_uses_systematic_sampling_l970_97078

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Systematic
  | Random
  | Stratified
  | Cluster

/-- Represents a production line with its characteristics --/
structure ProductionLine where
  daily_production : ℕ
  sampling_frequency : ℕ  -- days per week
  samples_per_day : ℕ
  sampling_start_time : ℕ  -- in minutes past midnight
  sampling_end_time : ℕ    -- in minutes past midnight

/-- Determines the sampling method based on production line characteristics --/
def determine_sampling_method (pl : ProductionLine) : SamplingMethod :=
  sorry  -- Proof to be implemented

/-- Theorem stating that the given production line uses systematic sampling --/
theorem production_line_uses_systematic_sampling (pl : ProductionLine) 
  (h1 : pl.daily_production = 128)
  (h2 : pl.sampling_frequency = 7)  -- weekly
  (h3 : pl.samples_per_day = 8)
  (h4 : pl.sampling_start_time = 14 * 60)  -- 2:00 PM
  (h5 : pl.sampling_end_time = 14 * 60 + 30)  -- 2:30 PM
  : determine_sampling_method pl = SamplingMethod.Systematic :=
by
  sorry  -- Proof to be implemented

end NUMINAMATH_CALUDE_production_line_uses_systematic_sampling_l970_97078


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l970_97020

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l970_97020


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l970_97062

theorem polynomial_subtraction (x : ℝ) :
  (4*x - 3) * (x + 6) - (2*x + 1) * (x - 4) = 2*x^2 + 28*x - 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l970_97062


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_inverse_l970_97070

def z : ℂ := 3 + Complex.I

theorem imaginary_part_of_z_plus_inverse (z : ℂ) (h : z = 3 + Complex.I) :
  Complex.im (z + z⁻¹) = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_inverse_l970_97070


namespace NUMINAMATH_CALUDE_binomial_prob_half_l970_97072

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_prob_half (ξ : BinomialRV) 
  (h_exp : expected_value ξ = 2)
  (h_var : variance ξ = 1) : 
  ξ.p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_half_l970_97072


namespace NUMINAMATH_CALUDE_total_pupils_l970_97076

theorem total_pupils (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 542) (h2 : boys = 387) : 
  girls + boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l970_97076


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l970_97050

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a ∧
  ((1/2) * (a + b)^2 + (1/4) * (a + b) = a * Real.sqrt b + b * Real.sqrt a ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1/4 ∧ b = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l970_97050


namespace NUMINAMATH_CALUDE_tree_growth_rate_l970_97001

theorem tree_growth_rate (h : ℝ) (initial_height : ℝ) (growth_period : ℕ) :
  initial_height = 4 →
  growth_period = 6 →
  initial_height + 6 * h = (initial_height + 4 * h) * (1 + 1/7) →
  h = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l970_97001


namespace NUMINAMATH_CALUDE_cubic_root_sum_square_l970_97010

theorem cubic_root_sum_square (a b c s : ℝ) : 
  (a^3 - 12*a^2 + 14*a - 1 = 0) →
  (b^3 - 12*b^2 + 14*b - 1 = 0) →
  (c^3 - 12*c^2 + 14*c - 1 = 0) →
  (s = Real.sqrt a + Real.sqrt b + Real.sqrt c) →
  (s^4 - 24*s^2 - 10*s = -144) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_square_l970_97010


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l970_97012

theorem tan_sum_specific_angles (α β : ℝ) 
  (h1 : 2 * Real.tan α = 1) 
  (h2 : Real.tan β = -2) : 
  Real.tan (α + β) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l970_97012


namespace NUMINAMATH_CALUDE_correct_number_of_fills_l970_97022

/-- The number of times Alice must fill her measuring cup -/
def number_of_fills : ℕ := 12

/-- The amount of sugar Alice needs in cups -/
def sugar_needed : ℚ := 15/4

/-- The capacity of Alice's measuring cup in cups -/
def cup_capacity : ℚ := 1/3

/-- Theorem stating that the number of fills is correct -/
theorem correct_number_of_fills :
  (↑number_of_fills : ℚ) * cup_capacity ≥ sugar_needed ∧
  ((↑number_of_fills - 1 : ℚ) * cup_capacity < sugar_needed) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_fills_l970_97022


namespace NUMINAMATH_CALUDE_mias_christmas_gifts_l970_97089

/-- Proves that the amount spent on each parent's gift is $30 -/
theorem mias_christmas_gifts (total_spent : ℕ) (sibling_gift : ℕ) (num_siblings : ℕ) :
  total_spent = 150 ∧ sibling_gift = 30 ∧ num_siblings = 3 →
  ∃ (parent_gift : ℕ), 
    parent_gift * 2 + sibling_gift * num_siblings = total_spent ∧
    parent_gift = 30 :=
by sorry

end NUMINAMATH_CALUDE_mias_christmas_gifts_l970_97089


namespace NUMINAMATH_CALUDE_ice_cubes_per_tray_l970_97071

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) 
  (h1 : total_ice_cubes = 72) 
  (h2 : number_of_trays = 8) 
  (h3 : total_ice_cubes % number_of_trays = 0) : 
  total_ice_cubes / number_of_trays = 9 := by
  sorry

end NUMINAMATH_CALUDE_ice_cubes_per_tray_l970_97071


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l970_97003

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a - b)^2 = 4 * (a * b)^3) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x - y)^2 = 4 * (x * y)^3 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l970_97003


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l970_97030

theorem unique_solution_for_equation : ∃! (n : ℕ), 
  ∃ (x : ℕ), x > 0 ∧ 
  n = 2^(2*x - 1) - 5*x - 3 ∧
  n = (2^(x-1) - 1) * (2^x + 1) ∧
  n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l970_97030


namespace NUMINAMATH_CALUDE_overtake_time_l970_97031

/-- The time it takes for a faster runner to overtake and finish ahead of a slower runner -/
theorem overtake_time (initial_distance steve_speed john_speed final_distance : ℝ) 
  (h1 : initial_distance = 12)
  (h2 : steve_speed = 3.7)
  (h3 : john_speed = 4.2)
  (h4 : final_distance = 2)
  (h5 : john_speed > steve_speed) :
  (initial_distance + final_distance) / (john_speed - steve_speed) = 28 := by
  sorry

#check overtake_time

end NUMINAMATH_CALUDE_overtake_time_l970_97031


namespace NUMINAMATH_CALUDE_fish_remaining_l970_97000

theorem fish_remaining (guppies angelfish tiger_sharks oscar_fish : ℕ)
  (guppies_sold angelfish_sold tiger_sharks_sold oscar_fish_sold : ℕ)
  (h1 : guppies = 94)
  (h2 : angelfish = 76)
  (h3 : tiger_sharks = 89)
  (h4 : oscar_fish = 58)
  (h5 : guppies_sold = 30)
  (h6 : angelfish_sold = 48)
  (h7 : tiger_sharks_sold = 17)
  (h8 : oscar_fish_sold = 24) :
  (guppies - guppies_sold) + (angelfish - angelfish_sold) +
  (tiger_sharks - tiger_sharks_sold) + (oscar_fish - oscar_fish_sold) = 198 :=
by sorry

end NUMINAMATH_CALUDE_fish_remaining_l970_97000


namespace NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l970_97082

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSideLength (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- Theorem: The smallest square area to fit 2×4 and 3×5 rectangles is 25 -/
theorem smallest_square_area_for_two_rectangles :
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  (minSquareSideLength r1 r2)^2 = 25 := by
  sorry

#eval (minSquareSideLength ⟨2, 4⟩ ⟨3, 5⟩)^2

end NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l970_97082


namespace NUMINAMATH_CALUDE_folded_paper_cut_ratio_l970_97025

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_cut_ratio :
  let original_side : ℝ := 6
  let folded_paper := Rectangle.mk original_side (original_side / 2)
  let large_rectangle := folded_paper
  let small_rectangle := Rectangle.mk (original_side / 2) (original_side / 2)
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_cut_ratio_l970_97025


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l970_97098

/-- Represents an ellipse with semi-major axis a and eccentricity e -/
structure Ellipse where
  a : ℝ
  e : ℝ

/-- The equation of the ellipse in terms of m -/
def ellipse_equation (m : ℝ) : Prop :=
  m > 1 ∧ ∃ x y : ℝ, x^2 / m^2 + y^2 / (m^2 - 1) = 1

/-- The distances from a point on the ellipse to its foci -/
def focus_distances (left right : ℝ) : Prop :=
  left = 3 ∧ right = 1

/-- The theorem stating the eccentricity of the ellipse -/
theorem ellipse_eccentricity (m : ℝ) :
  ellipse_equation m →
  (∃ left right : ℝ, focus_distances left right) →
  ∃ e : Ellipse, e.e = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l970_97098


namespace NUMINAMATH_CALUDE_log_8_x_equals_3_5_l970_97015

theorem log_8_x_equals_3_5 (x : ℝ) : 
  Real.log x / Real.log 8 = 3.5 → x = 181.04 := by
  sorry

end NUMINAMATH_CALUDE_log_8_x_equals_3_5_l970_97015
