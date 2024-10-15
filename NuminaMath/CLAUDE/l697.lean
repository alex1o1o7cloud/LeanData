import Mathlib

namespace NUMINAMATH_CALUDE_integer_triple_product_sum_l697_69763

theorem integer_triple_product_sum (a b c : ℤ) : 
  (a * b * c = 4 * (a + b + c) ∧ c = 2 * (a + b)) ↔ 
  ((a = 1 ∧ b = 6 ∧ c = 14) ∨ 
   (a = -1 ∧ b = -6 ∧ c = -14) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 10) ∨ 
   (a = -2 ∧ b = -3 ∧ c = -10) ∨ 
   (b = -a ∧ c = 0)) := by
sorry

end NUMINAMATH_CALUDE_integer_triple_product_sum_l697_69763


namespace NUMINAMATH_CALUDE_min_value_expression_l697_69726

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l697_69726


namespace NUMINAMATH_CALUDE_five_letter_words_same_ends_l697_69760

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word --/
def free_letters : ℕ := word_length - 2

/-- The number of five-letter words with the same first and last letter --/
def count_words : ℕ := alphabet_size ^ (free_letters + 1)

theorem five_letter_words_same_ends :
  count_words = 456976 := by sorry

end NUMINAMATH_CALUDE_five_letter_words_same_ends_l697_69760


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l697_69743

def is_odd (n : ℤ) : Prop := ∃ k, n = 2*k + 1

def is_even (n : ℤ) : Prop := ∃ k, n = 2*k

theorem contrapositive_odd_sum_even :
  (∀ a b : ℤ, (is_odd a ∧ is_odd b) → is_even (a + b)) ↔
  (∀ a b : ℤ, ¬is_even (a + b) → ¬(is_odd a ∧ is_odd b)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l697_69743


namespace NUMINAMATH_CALUDE_angle_measure_l697_69777

/-- An angle in degrees satisfies the given condition if its supplement is four times its complement -/
theorem angle_measure (x : ℝ) : (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l697_69777


namespace NUMINAMATH_CALUDE_fruit_distribution_l697_69742

/-- The number of ways to distribute n identical items among k distinct recipients --/
def distribute_identical (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute m distinct items among k distinct recipients --/
def distribute_distinct (m k : ℕ) : ℕ := k^m

theorem fruit_distribution :
  let apples : ℕ := 6
  let distinct_fruits : ℕ := 3  -- orange, plum, tangerine
  let people : ℕ := 3
  distribute_identical apples people * distribute_distinct distinct_fruits people = 756 := by
sorry

end NUMINAMATH_CALUDE_fruit_distribution_l697_69742


namespace NUMINAMATH_CALUDE_senior_teachers_in_sample_l697_69795

theorem senior_teachers_in_sample
  (total_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (sample_intermediate : ℕ)
  (h_total : total_teachers = 300)
  (h_intermediate : intermediate_teachers = 192)
  (h_sample_intermediate : sample_intermediate = 64)
  (h_ratio : ∃ k : ℕ, k > 0 ∧ total_teachers - intermediate_teachers = 9 * k ∧ 5 * k = 4 * k + k) :
  ∃ sample_size : ℕ,
    sample_size * intermediate_teachers = sample_intermediate * total_teachers ∧
    ∃ sample_senior : ℕ,
      9 * sample_senior = 5 * (sample_size - sample_intermediate) ∧
      sample_senior = 20 :=
sorry

end NUMINAMATH_CALUDE_senior_teachers_in_sample_l697_69795


namespace NUMINAMATH_CALUDE_triangle_angle_with_sine_half_l697_69755

theorem triangle_angle_with_sine_half (α : Real) :
  0 < α ∧ α < π ∧ Real.sin α = 1/2 → α = π/6 ∨ α = 5*π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_with_sine_half_l697_69755


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_two_l697_69711

theorem solution_set_abs_x_times_x_minus_two (x : ℝ) :
  {x : ℝ | |x| * (x - 2) ≥ 0} = {x : ℝ | x ≥ 2 ∨ x = 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_two_l697_69711


namespace NUMINAMATH_CALUDE_parrots_per_cage_l697_69725

theorem parrots_per_cage (num_cages : ℝ) (parakeets_per_cage : ℝ) (total_birds : ℕ) :
  num_cages = 6 →
  parakeets_per_cage = 2 →
  total_birds = 48 →
  ∃ parrots_per_cage : ℕ, 
    (parrots_per_cage : ℝ) * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 6 := by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l697_69725


namespace NUMINAMATH_CALUDE_pizza_combinations_l697_69767

/-- The number of available toppings -/
def num_toppings : ℕ := 8

/-- The number of forbidden topping combinations -/
def num_forbidden : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of possible one- and two-topping pizzas -/
def total_pizzas : ℕ := num_toppings + choose num_toppings 2 - num_forbidden

theorem pizza_combinations :
  total_pizzas = 35 :=
sorry

end NUMINAMATH_CALUDE_pizza_combinations_l697_69767


namespace NUMINAMATH_CALUDE_min_attempts_eq_n_l697_69720

/-- Represents a binary code of length n -/
def BinaryCode (n : ℕ) := Fin n → Bool

/-- Feedback from an attempt -/
inductive Feedback
| NoClick
| Click
| Open

/-- Function representing an attempt to open the safe -/
def attempt (n : ℕ) (secretCode : BinaryCode n) (tryCode : BinaryCode n) : Feedback :=
  sorry

/-- Minimum number of attempts required to open the safe -/
def minAttempts (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of attempts is n -/
theorem min_attempts_eq_n (n : ℕ) : minAttempts n = n :=
  sorry

end NUMINAMATH_CALUDE_min_attempts_eq_n_l697_69720


namespace NUMINAMATH_CALUDE_f_1998_is_zero_l697_69790

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_1998_is_zero
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 1998 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_1998_is_zero_l697_69790


namespace NUMINAMATH_CALUDE_caz_at_position_p_l697_69770

-- Define the type for positions in the gallery
inductive Position
| P
| Other

-- Define the type for people in the gallery
inductive Person
| Ali
| Bea
| Caz
| Dan

-- Define the visibility relation
def CanSee (a b : Person) : Prop := sorry

-- Define the position of a person
def IsAt (p : Person) (pos : Position) : Prop := sorry

-- State the theorem
theorem caz_at_position_p :
  -- Conditions
  (∀ x, x ≠ Person.Ali → ¬CanSee Person.Ali x) →
  (CanSee Person.Bea Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Bea x) →
  (CanSee Person.Caz Person.Bea) →
  (CanSee Person.Caz Person.Dan) →
  (∀ x, x ≠ Person.Bea ∧ x ≠ Person.Dan → ¬CanSee Person.Caz x) →
  (CanSee Person.Dan Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Dan x) →
  -- Conclusion
  IsAt Person.Caz Position.P :=
by sorry

end NUMINAMATH_CALUDE_caz_at_position_p_l697_69770


namespace NUMINAMATH_CALUDE_one_belt_one_road_values_road_line_equation_l697_69749

/-- Definition of "one belt, one road" relationship -/
def one_belt_one_road (a b c m n : ℝ) : Prop :=
  ∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ y = m * x + 1 ∧
  (∃ (x₀ : ℝ), a * x₀^2 + b * x₀ + c = m * x₀ + 1 ∧
   ∀ (x : ℝ), a * x^2 + b * x + c ≥ m * x + 1)

/-- Theorem for part 1 -/
theorem one_belt_one_road_values :
  one_belt_one_road 1 (-2) n (-1) 1 :=
sorry

/-- Theorem for part 2 -/
theorem road_line_equation (m n : ℝ) :
  (∃ (x : ℝ), m * (x + 1)^2 - 6 = 6 / x) ∧
  (∀ (x : ℝ), m * (x + 1)^2 - 6 ≥ 2 * x - 4) →
  m = 2 ∨ m = -2/3 :=
sorry

end NUMINAMATH_CALUDE_one_belt_one_road_values_road_line_equation_l697_69749


namespace NUMINAMATH_CALUDE_solution_for_x_and_y_l697_69712

theorem solution_for_x_and_y (a x y : Real) (k : Int) (h1 : x + y = a) (h2 : Real.sin x ^ 2 + Real.sin y ^ 2 = 1 - Real.cos a) (h3 : Real.cos a ≠ 0) :
  x = a / 2 + k * Real.pi ∧ y = a / 2 - k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_solution_for_x_and_y_l697_69712


namespace NUMINAMATH_CALUDE_frog_population_estimate_l697_69710

/-- Estimates the number of frogs in a pond based on capture-recapture data and population changes --/
theorem frog_population_estimate (tagged_april : ℕ) (caught_august : ℕ) (tagged_recaptured : ℕ)
  (left_pond_percent : ℚ) (new_frogs_percent : ℚ)
  (h1 : tagged_april = 100)
  (h2 : caught_august = 90)
  (h3 : tagged_recaptured = 5)
  (h4 : left_pond_percent = 30 / 100)
  (h5 : new_frogs_percent = 35 / 100) :
  let april_frogs_in_august := caught_august * (1 - new_frogs_percent)
  let estimated_april_population := (tagged_april * april_frogs_in_august) / tagged_recaptured
  estimated_april_population = 1180 := by
sorry

end NUMINAMATH_CALUDE_frog_population_estimate_l697_69710


namespace NUMINAMATH_CALUDE_cube_root_of_negative_twenty_seven_l697_69729

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cube_root_of_negative_twenty_seven :
  cubeRoot (-27) = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_twenty_seven_l697_69729


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l697_69764

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 4*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 4*x + 4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l697_69764


namespace NUMINAMATH_CALUDE_range_of_m_l697_69776

-- Define set A
def A : Set ℝ := {y | ∃ x > 0, y = 1 / x}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 * x - 4)}

-- Theorem statement
theorem range_of_m (m : ℝ) (h1 : m ∈ A) (h2 : m ∉ B) : m ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l697_69776


namespace NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l697_69781

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- Calculate the lateral surface area of a regular square pyramid -/
def lateral_surface_area (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a regular square pyramid with 
    base edge length 2√2 cm and volume 8 cm³ is 4√22 cm² -/
theorem pyramid_lateral_surface_area :
  let p : RegularSquarePyramid := ⟨2 * Real.sqrt 2, 8⟩
  lateral_surface_area p = 4 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l697_69781


namespace NUMINAMATH_CALUDE_smallest_n_years_for_90_percent_depreciation_l697_69750

-- Define the depreciation rate
def depreciation_rate : ℝ := 0.9

-- Define the target depreciation
def target_depreciation : ℝ := 0.1

-- Define the approximation of log 3
def log3_approx : ℝ := 0.477

-- Define the function to check if n years of depreciation meets the target
def meets_target (n : ℕ) : Prop := depreciation_rate ^ n ≤ target_depreciation

-- Statement to prove
theorem smallest_n_years_for_90_percent_depreciation :
  ∃ n : ℕ, meets_target n ∧ ∀ m : ℕ, m < n → ¬meets_target m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_years_for_90_percent_depreciation_l697_69750


namespace NUMINAMATH_CALUDE_sum_of_squares_l697_69780

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + c * a = 6) (h2 : a + b + c = 15) :
  a^2 + b^2 + c^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l697_69780


namespace NUMINAMATH_CALUDE_rabbit_count_l697_69779

/-- Given a cage with chickens and rabbits, prove that the number of rabbits is 31 -/
theorem rabbit_count (total_heads : ℕ) (r c : ℕ) : 
  total_heads = 51 →
  r + c = total_heads →
  4 * r = 3 * (2 * c) + 4 →
  r = 31 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l697_69779


namespace NUMINAMATH_CALUDE_quadratic_sum_l697_69791

/-- A quadratic function g(x) = px^2 + qx + r passing through (0, 3) and (2, 3) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  QuadraticFunction p q r 0 = 3 ∧ QuadraticFunction p q r 2 = 3 →
  p + 2*q + r = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l697_69791


namespace NUMINAMATH_CALUDE_function_composition_equals_log_l697_69765

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1/2 * x - 1/2 else Real.log x

theorem function_composition_equals_log (a : ℝ) :
  (f (f a) = Real.log (f a)) ↔ a ∈ Set.Ici (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_function_composition_equals_log_l697_69765


namespace NUMINAMATH_CALUDE_x_seventh_minus_27x_squared_l697_69741

theorem x_seventh_minus_27x_squared (x : ℝ) (h : x^3 - 3*x = 6) :
  x^7 - 27*x^2 = 9*(x + 1)*(x + 6) := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_minus_27x_squared_l697_69741


namespace NUMINAMATH_CALUDE_derivative_positive_solution_set_l697_69748

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def solution_set : Set ℝ := Set.Ioi 2

theorem derivative_positive_solution_set :
  ∀ x > 0, x ∈ solution_set ↔ deriv f x > 0 :=
sorry

end NUMINAMATH_CALUDE_derivative_positive_solution_set_l697_69748


namespace NUMINAMATH_CALUDE_marbles_combination_l697_69752

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marbles_combination :
  choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_marbles_combination_l697_69752


namespace NUMINAMATH_CALUDE_delivery_driver_net_pay_l697_69708

/-- Calculates the net rate of pay for a delivery driver --/
theorem delivery_driver_net_pay 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) 
  (gasoline_price : ℝ) 
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : earnings_per_mile = 0.60)
  (h5 : gasoline_price = 2.50) : 
  (earnings_per_mile * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_price) / travel_time = 25 := by
  sorry

#check delivery_driver_net_pay

end NUMINAMATH_CALUDE_delivery_driver_net_pay_l697_69708


namespace NUMINAMATH_CALUDE_hyperbola_equation_l697_69728

/-- The standard equation of a hyperbola with given focus and conjugate axis endpoint -/
theorem hyperbola_equation (f : ℝ × ℝ) (e : ℝ × ℝ) :
  f = (-10, 0) →
  e = (0, 4) →
  ∀ x y : ℝ, (x^2 / 84 - y^2 / 16 = 1) ↔ 
    (∃ a b c : ℝ, a^2 = 84 ∧ b^2 = 16 ∧ c^2 = a^2 + b^2 ∧
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      c = 10 ∧ 
      (x - f.1)^2 + (y - f.2)^2 - ((x + 10)^2 + y^2) = 4 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l697_69728


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l697_69773

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 10) / p = (q^3 + 10) / q ∧ (q^3 + 10) / q = (r^3 + 10) / r) : 
  p^3 + q^3 + r^3 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l697_69773


namespace NUMINAMATH_CALUDE_binomial_60_3_l697_69766

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l697_69766


namespace NUMINAMATH_CALUDE_expression_values_l697_69732

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l697_69732


namespace NUMINAMATH_CALUDE_angle_between_vectors_is_pi_over_3_l697_69701

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors_is_pi_over_3 (a b : ℝ × ℝ) 
  (h1 : a • (a + b) = 5)
  (h2 : ‖a‖ = 2)
  (h3 : ‖b‖ = 1) : 
  angle_between_vectors a b = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_is_pi_over_3_l697_69701


namespace NUMINAMATH_CALUDE_alvarez_diesel_consumption_l697_69796

/-- Given that Mr. Alvarez spends $36 on diesel fuel each week and the cost of diesel fuel is $3 per gallon,
    prove that he uses 24 gallons of diesel fuel in two weeks. -/
theorem alvarez_diesel_consumption
  (weekly_expenditure : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_expenditure = 36)
  (h2 : cost_per_gallon = 3)
  : (weekly_expenditure / cost_per_gallon) * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alvarez_diesel_consumption_l697_69796


namespace NUMINAMATH_CALUDE_reciprocal_of_five_eighths_l697_69717

theorem reciprocal_of_five_eighths :
  let x : ℚ := 5 / 8
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_five_eighths_l697_69717


namespace NUMINAMATH_CALUDE_sequence_equality_l697_69719

/-- Given two sequences of real numbers (xₙ) and (yₙ) defined as follows:
    x₁ = y₁ = 1
    xₙ₊₁ = (xₙ + 2) / (xₙ + 1)
    yₙ₊₁ = (yₙ² + 2) / (2yₙ)
    Prove that yₙ₊₁ = x₂ⁿ holds for n = 0, 1, 2, ... -/
theorem sequence_equality (x y : ℕ → ℝ) 
    (hx1 : x 1 = 1)
    (hy1 : y 1 = 1)
    (hx : ∀ n : ℕ, x (n + 1) = (x n + 2) / (x n + 1))
    (hy : ∀ n : ℕ, y (n + 1) = (y n ^ 2 + 2) / (2 * y n)) :
  ∀ n : ℕ, y (n + 1) = x (2 ^ n) := by
  sorry


end NUMINAMATH_CALUDE_sequence_equality_l697_69719


namespace NUMINAMATH_CALUDE_bowling_team_new_average_l697_69702

def bowling_team_average (original_players : ℕ) (original_average : ℚ) (new_player1_weight : ℚ) (new_player2_weight : ℚ) : ℚ :=
  let original_total_weight := original_players * original_average
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

theorem bowling_team_new_average :
  bowling_team_average 7 121 110 60 = 113 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_new_average_l697_69702


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l697_69724

theorem shaded_area_theorem (total_area : ℝ) (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_area = 64 → 
  total_triangles = 64 → 
  shaded_triangles = 28 → 
  (shaded_triangles : ℝ) * (total_area / total_triangles) = 28 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l697_69724


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l697_69713

theorem definite_integral_exp_plus_2x : ∫ (x : ℝ) in (0)..(1), (Real.exp x + 2 * x) = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l697_69713


namespace NUMINAMATH_CALUDE_stating_number_of_regions_correct_l697_69709

/-- 
Given n lines in a plane where no two lines are parallel and no three lines are concurrent,
this function returns the number of regions the plane is divided into.
-/
def number_of_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- 
Theorem stating that n lines in a plane, with no two lines parallel and no three lines concurrent,
divide the plane into (n(n+1)/2) + 1 regions.
-/
theorem number_of_regions_correct (n : ℕ) : 
  number_of_regions n = n * (n + 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_stating_number_of_regions_correct_l697_69709


namespace NUMINAMATH_CALUDE_unsold_bars_unsold_bars_correct_l697_69757

/-- Proves the number of unsold chocolate bars given the total number of bars,
    the cost per bar, and the total money made from sold bars. -/
theorem unsold_bars (total_bars : ℕ) (cost_per_bar : ℕ) (money_made : ℕ) : ℕ :=
  total_bars - (money_made / cost_per_bar)

#check unsold_bars 7 3 9 = 4

theorem unsold_bars_correct :
  unsold_bars 7 3 9 = 4 := by sorry

end NUMINAMATH_CALUDE_unsold_bars_unsold_bars_correct_l697_69757


namespace NUMINAMATH_CALUDE_custom_op_theorem_l697_69775

/-- Custom operation ã — -/
def custom_op (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem custom_op_theorem :
  ∃ X : ℝ, X + 2 * (custom_op 1 3) = 7 →
  3 * (custom_op 1 2) = 12 * 1 - 18 := by
sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l697_69775


namespace NUMINAMATH_CALUDE_third_meeting_at_45km_l697_69738

/-- Two people moving with constant speeds on a 100 km path between points A and B -/
structure TwoMovers :=
  (speed_ratio : ℚ)
  (first_meet : ℚ)
  (second_meet : ℚ)

/-- The third meeting point of two movers given their speed ratio and first two meeting points -/
def third_meeting_point (m : TwoMovers) : ℚ :=
  100 - (3 / 8) * 200

/-- Theorem stating that under given conditions, the third meeting point is 45 km from A -/
theorem third_meeting_at_45km (m : TwoMovers) 
  (h1 : m.first_meet = 20)
  (h2 : m.second_meet = 80)
  (h3 : m.speed_ratio = 3 / 5) :
  third_meeting_point m = 45 := by
  sorry

#eval third_meeting_point { speed_ratio := 3 / 5, first_meet := 20, second_meet := 80 }

end NUMINAMATH_CALUDE_third_meeting_at_45km_l697_69738


namespace NUMINAMATH_CALUDE_price_calculation_l697_69704

/-- Calculates the total price for jewelry and paintings after a price increase -/
def total_price (
  original_jewelry_price : ℕ
  ) (original_painting_price : ℕ
  ) (jewelry_price_increase : ℕ
  ) (painting_price_increase_percent : ℕ
  ) (jewelry_quantity : ℕ
  ) (painting_quantity : ℕ
  ) : ℕ :=
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percent) / 100
  (new_jewelry_price * jewelry_quantity) + (new_painting_price * painting_quantity)

theorem price_calculation :
  total_price 30 100 10 20 2 5 = 680 := by
  sorry

end NUMINAMATH_CALUDE_price_calculation_l697_69704


namespace NUMINAMATH_CALUDE_hcf_problem_l697_69735

theorem hcf_problem (x y : ℕ+) 
  (h1 : Nat.lcm x y = 560) 
  (h2 : x * y = 42000) : 
  Nat.gcd x y = 75 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l697_69735


namespace NUMINAMATH_CALUDE_cost_difference_l697_69723

def ice_cream_cartons : ℕ := 100
def yoghurt_cartons : ℕ := 35
def ice_cream_cost_per_carton : ℚ := 12
def yoghurt_cost_per_carton : ℚ := 3
def ice_cream_discount_rate : ℚ := 0.05
def yoghurt_tax_rate : ℚ := 0.08

def ice_cream_total_cost : ℚ := ice_cream_cartons * ice_cream_cost_per_carton
def yoghurt_total_cost : ℚ := yoghurt_cartons * yoghurt_cost_per_carton

def ice_cream_discounted_cost : ℚ := ice_cream_total_cost * (1 - ice_cream_discount_rate)
def yoghurt_taxed_cost : ℚ := yoghurt_total_cost * (1 + yoghurt_tax_rate)

theorem cost_difference : 
  ice_cream_discounted_cost - yoghurt_taxed_cost = 1026.60 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l697_69723


namespace NUMINAMATH_CALUDE_wilson_children_ages_l697_69778

theorem wilson_children_ages (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_youngest : a = 4) (h_middle : b = 7) (h_average : (a + b + c) / 3 = 7) :
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_wilson_children_ages_l697_69778


namespace NUMINAMATH_CALUDE_cricket_game_initial_overs_l697_69787

/-- Prove that the number of overs played initially is 10, given the conditions of the cricket game. -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
    target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_game_initial_overs_l697_69787


namespace NUMINAMATH_CALUDE_pet_store_cages_l697_69747

/-- Given a pet store scenario with puppies and cages, calculate the number of cages used. -/
theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : puppies_per_cage = 2)
  : (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l697_69747


namespace NUMINAMATH_CALUDE_negative_sqrt_eleven_squared_l697_69727

theorem negative_sqrt_eleven_squared : (-Real.sqrt 11)^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_eleven_squared_l697_69727


namespace NUMINAMATH_CALUDE_cubic_sum_product_l697_69784

theorem cubic_sum_product (a b c : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a^2 + b^2 + c^2 = 15)
  (h3 : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_product_l697_69784


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l697_69718

theorem partial_fraction_decomposition (x A B C : ℝ) :
  x ≠ 2 → x ≠ 4 →
  (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) ↔
  (A = 20 ∧ B = -15 ∧ C = -10) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l697_69718


namespace NUMINAMATH_CALUDE_building_height_percentage_l697_69734

theorem building_height_percentage (L M R : ℝ) : 
  M = 100 → 
  R = L + M - 20 → 
  L + M + R = 340 → 
  L / M * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_building_height_percentage_l697_69734


namespace NUMINAMATH_CALUDE_smallest_divisible_number_after_2013_l697_69703

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 ∧ i < 10 → n % i = 0

theorem smallest_divisible_number_after_2013 :
  ∃ (n : ℕ),
    n ≥ 2013000 ∧
    is_divisible_by_all_less_than_10 n ∧
    (∀ m : ℕ, 2013000 ≤ m ∧ m < n → ¬is_divisible_by_all_less_than_10 m) ∧
    n = 2013480 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_after_2013_l697_69703


namespace NUMINAMATH_CALUDE_solution_set_l697_69740

theorem solution_set (x y : ℝ) : 
  x - 2*y = 1 → x^3 - 8*y^3 - 6*x*y = 1 → y = (x-1)/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l697_69740


namespace NUMINAMATH_CALUDE_swap_positions_l697_69792

/-- Represents the color of a checker -/
inductive Color
| Black
| White

/-- Represents a move in the game -/
structure Move where
  color : Color
  count : Nat

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  blackPositions : List Nat
  whitePositions : List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move.color with
  | Color.Black => move.count ≤ state.n ∧ move.count > 0
  | Color.White => move.count ≤ state.n ∧ move.count > 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Generates the sequence of moves for the game -/
def generateMoves (n : Nat) : List Move :=
  sorry

/-- Checks if the final state has swapped positions -/
def isSwappedState (initialState : GameState) (finalState : GameState) : Prop :=
  sorry

/-- Theorem stating that the generated moves will swap the positions -/
theorem swap_positions (n : Nat) :
  let initialState : GameState := { n := n, blackPositions := List.range n, whitePositions := List.range n |>.map (λ x => 2*n - x) }
  let moves := generateMoves n
  let finalState := moves.foldl applyMove initialState
  (∀ move ∈ moves, isValidMove initialState move) ∧
  isSwappedState initialState finalState :=
sorry

end NUMINAMATH_CALUDE_swap_positions_l697_69792


namespace NUMINAMATH_CALUDE_constant_term_value_l697_69788

theorem constant_term_value (x y z : ℤ) (k : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = k)
  (h_x : x = 20) : k = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l697_69788


namespace NUMINAMATH_CALUDE_largest_root_range_l697_69774

theorem largest_root_range (b₀ b₁ b₂ : ℝ) 
  (h₀ : |b₀| < 3) (h₁ : |b₁| < 3) (h₂ : |b₂| < 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 5 ∧
  (∀ x : ℝ, x > 0 → x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) ∧
  (∃ x : ℝ, x > 0 ∧ x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 ∧ x = r) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_range_l697_69774


namespace NUMINAMATH_CALUDE_flower_percentages_l697_69722

def total_flowers : ℕ := 30
def red_flowers : ℕ := 7
def white_flowers : ℕ := 6
def blue_flowers : ℕ := 5
def yellow_flowers : ℕ := 4

def purple_flowers : ℕ := total_flowers - (red_flowers + white_flowers + blue_flowers + yellow_flowers)

def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem flower_percentages :
  (percentage (red_flowers + white_flowers + blue_flowers) total_flowers = 60) ∧
  (percentage purple_flowers total_flowers = 26.67) ∧
  (percentage yellow_flowers total_flowers = 13.33) :=
by sorry

end NUMINAMATH_CALUDE_flower_percentages_l697_69722


namespace NUMINAMATH_CALUDE_max_planes_in_hangar_l697_69798

def hangar_length : ℕ := 300
def plane_length : ℕ := 40

theorem max_planes_in_hangar :
  (hangar_length / plane_length : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_in_hangar_l697_69798


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l697_69733

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l697_69733


namespace NUMINAMATH_CALUDE_water_added_to_container_l697_69731

/-- The amount of water added to fill a container from 30% to 3/4 full -/
theorem water_added_to_container (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 100)
  (h2 : initial_fraction = 0.3)
  (h3 : final_fraction = 3/4) :
  final_fraction * capacity - initial_fraction * capacity = 45 :=
by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l697_69731


namespace NUMINAMATH_CALUDE_fraction_simplification_l697_69759

theorem fraction_simplification :
  1 / (1 / (1/2)^1 + 1 / (1/2)^2 + 1 / (1/2)^3) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l697_69759


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_three_lines_l697_69714

/-- A circle with center (0, k) where k > 8 is tangent to y = x, y = -x, and y = 8. Its radius is 8√2. -/
theorem circle_radius_tangent_to_three_lines (k : ℝ) (h1 : k > 8) : 
  let center := (0, k)
  let radius := (λ p : ℝ × ℝ ↦ Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2))
  let tangent_to_line := (λ l : ℝ × ℝ → Prop ↦ ∃ p, l p ∧ radius p = radius center)
  tangent_to_line (λ p ↦ p.2 = p.1) ∧ 
  tangent_to_line (λ p ↦ p.2 = -p.1) ∧
  tangent_to_line (λ p ↦ p.2 = 8) →
  radius center = 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_three_lines_l697_69714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l697_69737

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 10) : 
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l697_69737


namespace NUMINAMATH_CALUDE_cafeteria_pies_l697_69700

/-- Given a cafeteria with total apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def pies_made (total_apples handed_out apples_per_pie : ℕ) : ℕ :=
  (total_apples - handed_out) / apples_per_pie

/-- Theorem: The cafeteria can make 9 pies with the given conditions. -/
theorem cafeteria_pies :
  pies_made 525 415 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l697_69700


namespace NUMINAMATH_CALUDE_total_snake_owners_l697_69794

theorem total_snake_owners (total : Nat) (only_dogs : Nat) (only_cats : Nat) (only_birds : Nat) (only_snakes : Nat)
  (cats_and_dogs : Nat) (birds_and_dogs : Nat) (birds_and_cats : Nat) (snakes_and_dogs : Nat) (snakes_and_cats : Nat)
  (snakes_and_birds : Nat) (cats_dogs_snakes : Nat) (cats_dogs_birds : Nat) (cats_birds_snakes : Nat)
  (dogs_birds_snakes : Nat) (all_four : Nat)
  (h1 : total = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four = 10) :
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + all_four = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_l697_69794


namespace NUMINAMATH_CALUDE_crow_eating_time_l697_69753

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def eating_time (fraction : ℚ) : ℚ :=
  7.5 / (1/4) * fraction

theorem crow_eating_time :
  eating_time (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_crow_eating_time_l697_69753


namespace NUMINAMATH_CALUDE_rounding_comparison_l697_69768

theorem rounding_comparison (a b : ℝ) : 
  (2.35 ≤ a ∧ a ≤ 2.44) → 
  (2.395 ≤ b ∧ b ≤ 2.404) → 
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x = y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x > y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x < y) :=
by sorry

end NUMINAMATH_CALUDE_rounding_comparison_l697_69768


namespace NUMINAMATH_CALUDE_octagon_perimeter_l697_69715

/-- An octagon is a polygon with 8 sides -/
def Octagon : Type := Unit

/-- The length of each side of the octagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of its sides -/
def perimeter (p : Octagon) : ℝ := 8 * side_length

theorem octagon_perimeter : 
  ∀ (o : Octagon), perimeter o = 24 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_l697_69715


namespace NUMINAMATH_CALUDE_scientific_notation_of_9560000_l697_69756

theorem scientific_notation_of_9560000 :
  9560000 = 9.56 * (10 : ℝ) ^ 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_9560000_l697_69756


namespace NUMINAMATH_CALUDE_largest_product_of_three_l697_69797

def S : Finset Int := {-5, -4, -1, 6, 7}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≤ a * b * c) →
  a * b * c = 140 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l697_69797


namespace NUMINAMATH_CALUDE_coconut_grove_solution_l697_69736

-- Define the problem parameters
def coconut_grove (x : ℝ) : Prop :=
  -- (x + 3) trees yield 60 nuts per year
  ∃ (yield1 : ℝ), yield1 = 60 * (x + 3) ∧
  -- x trees yield 120 nuts per year
  ∃ (yield2 : ℝ), yield2 = 120 * x ∧
  -- (x - 3) trees yield 180 nuts per year
  ∃ (yield3 : ℝ), yield3 = 180 * (x - 3) ∧
  -- The average yield per year per tree is 100
  (yield1 + yield2 + yield3) / (3 * x) = 100

-- Theorem stating that x = 6 is the unique solution
theorem coconut_grove_solution :
  ∃! x : ℝ, coconut_grove x ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_coconut_grove_solution_l697_69736


namespace NUMINAMATH_CALUDE_golden_ratio_greater_than_three_fifths_l697_69783

theorem golden_ratio_greater_than_three_fifths : (Real.sqrt 5 - 1) / 2 > 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_greater_than_three_fifths_l697_69783


namespace NUMINAMATH_CALUDE_jones_clothes_count_l697_69793

/-- Represents the ratio of shirts to pants -/
def shirt_to_pants_ratio : ℕ := 6

/-- Represents the number of pants Mr. Jones owns -/
def pants_count : ℕ := 40

/-- Calculates the total number of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * pants_count + pants_count

/-- Proves that the total number of clothes Mr. Jones owns is 280 -/
theorem jones_clothes_count : total_clothes = 280 := by
  sorry

end NUMINAMATH_CALUDE_jones_clothes_count_l697_69793


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l697_69705

/-- A line that forms an isosceles right-angled triangle with the coordinate axes -/
structure IsoscelesRightTriangleLine where
  a : ℝ
  eq : (x y : ℝ) → Prop
  passes_through : eq 2 3
  isosceles_right : ∀ (x y : ℝ), eq x y → (x / a + y / a = 1) ∨ (x / a + y / (-a) = 1)

/-- The equation of the line is either x + y - 5 = 0 or x - y + 1 = 0 -/
theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) :
  (∀ x y, l.eq x y ↔ x + y - 5 = 0) ∨ (∀ x y, l.eq x y ↔ x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l697_69705


namespace NUMINAMATH_CALUDE_kaleb_net_profit_l697_69771

/-- Calculates the net profit for Kaleb's lawn mowing business --/
def net_profit (small_charge medium_charge large_charge : ℕ)
                (spring_small spring_medium spring_large : ℕ)
                (summer_small summer_medium summer_large : ℕ)
                (fuel_expense supply_cost : ℕ) : ℕ :=
  let spring_earnings := small_charge * spring_small + medium_charge * spring_medium + large_charge * spring_large
  let summer_earnings := small_charge * summer_small + medium_charge * summer_medium + large_charge * summer_large
  let total_earnings := spring_earnings + summer_earnings
  let total_lawns := spring_small + spring_medium + spring_large + summer_small + summer_medium + summer_large
  let total_expenses := fuel_expense * total_lawns + supply_cost
  total_earnings - total_expenses

theorem kaleb_net_profit :
  net_profit 10 20 30 2 3 1 10 8 5 2 60 = 402 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_net_profit_l697_69771


namespace NUMINAMATH_CALUDE_dylan_ice_cube_trays_l697_69745

/-- The number of ice cube trays Dylan needs to fill -/
def num_trays_to_fill (glass_cubes : ℕ) (pitcher_multiplier : ℕ) (tray_capacity : ℕ) : ℕ :=
  ((glass_cubes + glass_cubes * pitcher_multiplier) + tray_capacity - 1) / tray_capacity

/-- Theorem stating that Dylan needs to fill 2 ice cube trays -/
theorem dylan_ice_cube_trays : 
  num_trays_to_fill 8 2 12 = 2 := by
  sorry

#eval num_trays_to_fill 8 2 12

end NUMINAMATH_CALUDE_dylan_ice_cube_trays_l697_69745


namespace NUMINAMATH_CALUDE_parents_disagree_tuition_increase_l697_69785

theorem parents_disagree_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 20 / 100) : 
  total_parents - (total_parents * agree_percentage).floor = 640 := by
sorry

end NUMINAMATH_CALUDE_parents_disagree_tuition_increase_l697_69785


namespace NUMINAMATH_CALUDE_field_width_l697_69761

/-- A rectangular field with length 7/5 of its width and perimeter 336 meters has a width of 70 meters -/
theorem field_width (w : ℝ) (h1 : w > 0) : 
  2 * (7/5 * w + w) = 336 → w = 70 := by
  sorry

end NUMINAMATH_CALUDE_field_width_l697_69761


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l697_69754

theorem sum_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℝ := -18
  let b : ℝ := 54
  let c : ℝ := -72
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l697_69754


namespace NUMINAMATH_CALUDE_pizza_area_difference_l697_69762

theorem pizza_area_difference : ∃ (N : ℝ), 
  (abs (N - 96) < 1) ∧ 
  (π * 7^2 = π * 5^2 * (1 + N / 100)) := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_difference_l697_69762


namespace NUMINAMATH_CALUDE_family_weight_problem_l697_69772

/-- Given a family with a grandmother, daughter, and child, prove their weights satisfy certain conditions and the combined weight of the daughter and child is 60 kg. -/
theorem family_weight_problem (grandmother daughter child : ℝ) : 
  grandmother + daughter + child = 110 →
  child = (1 / 5) * grandmother →
  daughter = 50 →
  daughter + child = 60 := by
sorry

end NUMINAMATH_CALUDE_family_weight_problem_l697_69772


namespace NUMINAMATH_CALUDE_square_area_is_400_l697_69758

-- Define the radius of the circles
def circle_radius : ℝ := 5

-- Define the side length of the square
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- Theorem: The area of the square is 400 square inches
theorem square_area_is_400 : square_side_length ^ 2 = 400 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_400_l697_69758


namespace NUMINAMATH_CALUDE_league_games_l697_69786

theorem league_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l697_69786


namespace NUMINAMATH_CALUDE_xavier_speed_increase_time_l697_69707

/-- Represents the journey of Xavier from p to q -/
structure Journey where
  initialSpeed : ℝ  -- Initial speed in km/h
  speedIncrease : ℝ  -- Speed increase in km/h
  totalDistance : ℝ  -- Total distance in km
  totalTime : ℝ  -- Total time in hours

/-- Calculates the time at which Xavier increases his speed -/
def timeOfSpeedIncrease (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that Xavier increases his speed after 24 minutes -/
theorem xavier_speed_increase_time (j : Journey) 
  (h1 : j.initialSpeed = 50)
  (h2 : j.speedIncrease = 10)
  (h3 : j.totalDistance = 52)
  (h4 : j.totalTime = 48 / 60) : 
  timeOfSpeedIncrease j = 24 / 60 := by
  sorry

end NUMINAMATH_CALUDE_xavier_speed_increase_time_l697_69707


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l697_69782

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l697_69782


namespace NUMINAMATH_CALUDE_equation_solution_l697_69746

theorem equation_solution :
  ∃! x : ℝ, (9 - 3*x) * (3^x) - (x - 2) * (x^2 - 5*x + 6) = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l697_69746


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l697_69799

/-- The number of distinct terms in the expansion of (a+b)(a+c+d+e+f) -/
def num_distinct_terms : ℕ := 9

/-- The first polynomial -/
def first_poly (a b : ℝ) : ℝ := a + b

/-- The second polynomial -/
def second_poly (a c d e f : ℝ) : ℝ := a + c + d + e + f

/-- Theorem stating that the number of distinct terms in the expansion is 9 -/
theorem expansion_distinct_terms 
  (a b c d e f : ℝ) : 
  num_distinct_terms = 9 := by sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l697_69799


namespace NUMINAMATH_CALUDE_rationalize_denominator_l697_69751

theorem rationalize_denominator :
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l697_69751


namespace NUMINAMATH_CALUDE_nested_circles_radius_l697_69744

theorem nested_circles_radius (A₁ A₂ : ℝ) : 
  A₁ > 0 → 
  A₂ > 0 → 
  (∃ d : ℝ, A₂ = A₁ + d ∧ A₁ + 2*A₂ = A₂ + d) → 
  A₁ + 2*A₂ = π * 5^2 → 
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_nested_circles_radius_l697_69744


namespace NUMINAMATH_CALUDE_stream_speed_l697_69730

theorem stream_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (downstream_wind : ℝ) 
  (upstream_wind : ℝ) 
  (h1 : downstream_distance = 110) 
  (h2 : upstream_distance = 85) 
  (h3 : downstream_time = 5) 
  (h4 : upstream_time = 6) 
  (h5 : downstream_wind = 3) 
  (h6 : upstream_wind = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed + downstream_wind) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed + upstream_wind) * upstream_time ∧ 
    stream_speed = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l697_69730


namespace NUMINAMATH_CALUDE_truck_speed_l697_69706

/-- Calculates the speed of a truck in kilometers per hour -/
theorem truck_speed (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 10) :
  (distance / time) * 3.6 = 216 := by
  sorry

#check truck_speed

end NUMINAMATH_CALUDE_truck_speed_l697_69706


namespace NUMINAMATH_CALUDE_power_of_two_equation_l697_69789

theorem power_of_two_equation (k : ℤ) : 
  2^1998 - 2^1997 - 2^1996 + 2^1995 = k * 2^1995 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l697_69789


namespace NUMINAMATH_CALUDE_local_max_condition_l697_69716

/-- The function f(x) = x(x-m)² has a local maximum at x = 1 if and only if m = 3 -/
theorem local_max_condition (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), x * (x - m)^2 ≤ 1 * (1 - m)^2) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_local_max_condition_l697_69716


namespace NUMINAMATH_CALUDE_income_percentage_difference_l697_69769

/-- Given the monthly incomes of A and B in ratio 5:2, C's monthly income of 15000,
    and A's annual income of 504000, prove that B's monthly income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_monthly B_monthly C_monthly : ℕ),
    C_monthly = 15000 →
    A_monthly * 12 = 504000 →
    A_monthly * 2 = B_monthly * 5 →
    (B_monthly - C_monthly) * 100 = C_monthly * 12 := by
  sorry

end NUMINAMATH_CALUDE_income_percentage_difference_l697_69769


namespace NUMINAMATH_CALUDE_set_operations_l697_69739

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {3, 4, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 4, 5, 7, 9}) ∧
  (A ∩ B = {3, 5}) ∧
  ({x | x ∈ A ∧ x ∉ B} = {1, 7, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l697_69739


namespace NUMINAMATH_CALUDE_not_multiple_of_121_l697_69721

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_121_l697_69721
