import Mathlib

namespace NUMINAMATH_CALUDE_probability_yellow_ball_l355_35547

def total_balls : ℕ := 5
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3

theorem probability_yellow_ball :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l355_35547


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l355_35544

-- Define the number of red cards the shopkeeper has
def total_red_cards : ℕ := 208

-- Define the number of cards in a standard deck
def cards_per_deck : ℕ := 52

-- Theorem stating the number of decks the shopkeeper has
theorem shopkeeper_decks : 
  total_red_cards / cards_per_deck = 4 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l355_35544


namespace NUMINAMATH_CALUDE_biographies_shelved_l355_35503

def total_books : ℕ := 46
def top_section_books : ℕ := 24
def western_novels : ℕ := 5

def bottom_section_books : ℕ := total_books - top_section_books

def mystery_books : ℕ := bottom_section_books / 2

theorem biographies_shelved :
  total_books - top_section_books - mystery_books - western_novels = 6 :=
by sorry

end NUMINAMATH_CALUDE_biographies_shelved_l355_35503


namespace NUMINAMATH_CALUDE_triangle_inequality_with_square_roots_l355_35509

/-- Given a triangle with sides a, b, and c, the sum of the square roots of the semiperimeter minus each side is less than or equal to the sum of the square roots of the sides. Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality_with_square_roots (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_square_roots_l355_35509


namespace NUMINAMATH_CALUDE_binomial_self_one_binomial_600_600_l355_35577

theorem binomial_self_one (n : ℕ) : Nat.choose n n = 1 := by sorry

theorem binomial_600_600 : Nat.choose 600 600 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_self_one_binomial_600_600_l355_35577


namespace NUMINAMATH_CALUDE_fraction_division_multiplication_l355_35560

theorem fraction_division_multiplication :
  (3 : ℚ) / 7 / 4 * 2 = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_multiplication_l355_35560


namespace NUMINAMATH_CALUDE_no_more_birds_can_join_l355_35546

/-- Represents the weight capacity of the fence in pounds -/
def fence_capacity : ℝ := 20

/-- Represents the weight of the first bird in pounds -/
def bird1_weight : ℝ := 2.5

/-- Represents the weight of the second bird in pounds -/
def bird2_weight : ℝ := 3.5

/-- Represents the number of additional birds that joined -/
def additional_birds : ℕ := 4

/-- Represents the weight of each additional bird in pounds -/
def additional_bird_weight : ℝ := 2.8

/-- Represents the weight of each new bird that might join in pounds -/
def new_bird_weight : ℝ := 3

/-- Calculates the total weight of birds currently on the fence -/
def current_weight : ℝ := bird1_weight + bird2_weight + additional_birds * additional_bird_weight

/-- Theorem stating that no more 3 lb birds can join the fence without exceeding its capacity -/
theorem no_more_birds_can_join : 
  ∀ n : ℕ, current_weight + n * new_bird_weight ≤ fence_capacity → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_more_birds_can_join_l355_35546


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l355_35590

theorem vector_magnitude_problem (m : ℝ) (a : ℝ × ℝ) :
  m > 0 → a = (m, 4) → ‖a‖ = 5 → m = 3 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l355_35590


namespace NUMINAMATH_CALUDE_direct_proportion_function_l355_35529

/-- A function that is directly proportional to 2x+3 and passes through the point (1, -5) -/
def f (x : ℝ) : ℝ := -2 * x - 3

theorem direct_proportion_function :
  (∃ k : ℝ, ∀ x, f x = k * (2 * x + 3)) ∧
  f 1 = -5 ∧
  (∀ x, f x = -2 * x - 3) ∧
  (f (5/2) = 2) := by
  sorry

#check direct_proportion_function

end NUMINAMATH_CALUDE_direct_proportion_function_l355_35529


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l355_35581

/-- Pentagon formed by cutting a right-angled triangular corner from a rectangle -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  parallel_sides : Bool
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∃ (p : Pentagon),
    p.side1 = 9 ∧ p.side2 = 16 ∧ p.side3 = 30 ∧ p.side4 = 40 ∧ p.side5 = 41 ∧
    p.parallel_sides = true ∧
    p.triangle_leg1 = 9 ∧ p.triangle_leg2 = 40 ∧
    pentagon_area p = 1020 := by
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l355_35581


namespace NUMINAMATH_CALUDE_battle_station_staffing_l355_35596

/-- The number of ways to assign n distinct objects to k distinct positions,
    where each position must be filled by exactly one object. -/
def permutations (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- The number of job openings -/
def num_jobs : ℕ := 6

/-- The number of suitable candidates -/
def num_candidates : ℕ := 15

theorem battle_station_staffing :
  permutations num_candidates num_jobs = 3276000 := by sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l355_35596


namespace NUMINAMATH_CALUDE_contrapositive_square_equality_l355_35528

theorem contrapositive_square_equality (a b : ℝ) : a^2 ≠ b^2 → a ≠ b := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_square_equality_l355_35528


namespace NUMINAMATH_CALUDE_base_conversion_l355_35568

/-- Given that the base 6 number 123₆ is equal to the base b number 203ᵦ,
    prove that the positive value of b is 2√6. -/
theorem base_conversion (b : ℝ) (h : b > 0) : 
  (1 * 6^2 + 2 * 6 + 3 : ℝ) = 2 * b^2 + 3 → b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l355_35568


namespace NUMINAMATH_CALUDE_conference_games_l355_35554

/-- The number of games in a complete season for a sports conference --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem stating the number of games in the specific conference setup --/
theorem conference_games : total_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l355_35554


namespace NUMINAMATH_CALUDE_polynomial_identity_l355_35533

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

def polynomial_equation : Prop :=
  ∀ x : ℝ, (1 + x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5

theorem polynomial_identity (h : polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅) :
  a₀ = 1 ∧ (a₀ / 1 + a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 21 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l355_35533


namespace NUMINAMATH_CALUDE_zoo_incident_final_counts_l355_35543

def wombat_count : ℕ := 9
def rhea_count : ℕ := 3
def porcupine_count : ℕ := 2

def carson_claw_per_wombat : ℕ := 4
def ava_claw_per_rhea : ℕ := 1
def liam_quill_per_porcupine : ℕ := 6

def carson_reduction_percent : ℚ := 25 / 100
def ava_reduction_percent : ℚ := 25 / 100
def liam_reduction_percent : ℚ := 50 / 100

def carson_initial_claws : ℕ := wombat_count * carson_claw_per_wombat
def ava_initial_claws : ℕ := rhea_count * ava_claw_per_rhea
def liam_initial_quills : ℕ := porcupine_count * liam_quill_per_porcupine

theorem zoo_incident_final_counts :
  (carson_initial_claws - Int.floor (↑carson_initial_claws * carson_reduction_percent) = 27) ∧
  (ava_initial_claws - Int.floor (↑ava_initial_claws * ava_reduction_percent) = 3) ∧
  (liam_initial_quills - Int.floor (↑liam_initial_quills * liam_reduction_percent) = 6) :=
by sorry

end NUMINAMATH_CALUDE_zoo_incident_final_counts_l355_35543


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l355_35579

theorem repeating_decimal_sum : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 : ℚ) / 10^3 + (234 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, b = (567 : ℚ) / 10^3 + (567 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, c = (891 : ℚ) / 10^3 + (891 : ℚ) / (999 * 10^n)) ∧
    a - b + c = 186 / 333 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l355_35579


namespace NUMINAMATH_CALUDE_zeros_properties_l355_35507

noncomputable def f (θ : ℝ) : ℝ := Real.sin (4 * θ) + Real.sin (3 * θ)

theorem zeros_properties (θ₁ θ₂ θ₃ : ℝ) 
  (h1 : 0 < θ₁ ∧ θ₁ < π) 
  (h2 : 0 < θ₂ ∧ θ₂ < π) 
  (h3 : 0 < θ₃ ∧ θ₃ < π) 
  (h4 : θ₁ ≠ θ₂ ∧ θ₁ ≠ θ₃ ∧ θ₂ ≠ θ₃) 
  (h5 : f θ₁ = 0) 
  (h6 : f θ₂ = 0) 
  (h7 : f θ₃ = 0) : 
  (θ₁ + θ₂ + θ₃ = 12 * π / 7) ∧ 
  (Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ = 1 / 8) ∧ 
  (Real.cos θ₁ + Real.cos θ₂ + Real.cos θ₃ = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_zeros_properties_l355_35507


namespace NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l355_35517

theorem abs_2x_minus_7_not_positive (x : ℚ) : |2*x - 7| ≤ 0 ↔ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l355_35517


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l355_35586

theorem pure_imaginary_square_root (a : ℝ) : 
  (∃ (b : ℝ), (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l355_35586


namespace NUMINAMATH_CALUDE_chestnut_picking_l355_35556

/-- The amount of chestnuts picked by Mary, Peter, and Lucy -/
theorem chestnut_picking (mary peter lucy : ℝ) 
  (h1 : mary = 2 * peter)  -- Mary picked twice as much as Peter
  (h2 : lucy = peter + 2)  -- Lucy picked 2 kg more than Peter
  (h3 : mary = 12)         -- Mary picked 12 kg
  : mary + peter + lucy = 26 := by
  sorry

#check chestnut_picking

end NUMINAMATH_CALUDE_chestnut_picking_l355_35556


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l355_35587

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l355_35587


namespace NUMINAMATH_CALUDE_mod_29_graph_intercepts_sum_l355_35599

theorem mod_29_graph_intercepts_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 29 ∧ y₀ < 29 ∧
  (∀ x : ℤ, (4 * x) % 29 = (5 * 0 - 1) % 29 ↔ x % 29 = x₀) ∧
  (∀ y : ℤ, (4 * 0) % 29 = (5 * y - 1) % 29 ↔ y % 29 = y₀) ∧
  x₀ + y₀ = 30 :=
by sorry

end NUMINAMATH_CALUDE_mod_29_graph_intercepts_sum_l355_35599


namespace NUMINAMATH_CALUDE_game_boxes_needed_l355_35567

theorem game_boxes_needed (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : 
  initial_games = 76 → sold_games = 46 → games_per_box = 5 → 
  (initial_games - sold_games) / games_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_boxes_needed_l355_35567


namespace NUMINAMATH_CALUDE_general_term_formula_l355_35548

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n - 2 * a (n + 1) + a (n + 2) = 0

/-- Theorem stating the general term formula for the sequence -/
theorem general_term_formula (a : ℕ+ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_initial1 : a 1 = 2) 
    (h_initial2 : a 2 = 4) : 
    ∀ n : ℕ+, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l355_35548


namespace NUMINAMATH_CALUDE_problem_solution_l355_35588

theorem problem_solution :
  (∃ (x y : ℝ),
    (x = (3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 ∧ x = 2) ∧
    (y = (1 + Real.sqrt 2 + Real.sqrt 3) * (1 + Real.sqrt 2 - Real.sqrt 3) ∧ y = 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l355_35588


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l355_35594

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: Given the conditions, prove that the man's speed against the current is 10 km/hr -/
theorem mans_speed_against_current :
  let speed_with_current : ℝ := 15
  let current_speed : ℝ := 2.5
  speed_against_current speed_with_current current_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l355_35594


namespace NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l355_35511

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l355_35511


namespace NUMINAMATH_CALUDE_square_minus_four_times_product_specific_calculation_l355_35510

theorem square_minus_four_times_product (a b : ℕ) : 
  (a + b) ^ 2 - 4 * a * b = (a - b) ^ 2 :=
by sorry

theorem specific_calculation : 
  (476 + 424) ^ 2 - 4 * 476 * 424 = 5776 :=
by sorry

end NUMINAMATH_CALUDE_square_minus_four_times_product_specific_calculation_l355_35510


namespace NUMINAMATH_CALUDE_power_of_power_l355_35514

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l355_35514


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l355_35536

def M : Finset ℕ := {1, 2, 3}
def N : Finset ℕ := {4, 5, 6}

def is_sum_even (x : ℕ) (y : ℕ) : Bool := Even (x + y)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (M.product N).filter (fun (x, y) => is_sum_even x y)

theorem probability_of_even_sum :
  (favorable_outcomes.card : ℚ) / ((M.card * N.card) : ℚ) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l355_35536


namespace NUMINAMATH_CALUDE_factory_production_l355_35527

/-- Calculates the total number of products produced by a factory in 5 days -/
def total_products_in_five_days (refrigerators_per_hour : ℕ) (coolers_difference : ℕ) (hours_per_day : ℕ) : ℕ :=
  let coolers_per_hour := refrigerators_per_hour + coolers_difference
  let products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_hours := 5 * hours_per_day
  products_per_hour * total_hours

/-- Theorem stating that the factory produces 11250 products in 5 days -/
theorem factory_production :
  total_products_in_five_days 90 70 9 = 11250 :=
by
  sorry

end NUMINAMATH_CALUDE_factory_production_l355_35527


namespace NUMINAMATH_CALUDE_quadratic_roots_l355_35553

theorem quadratic_roots (p q : ℚ) : 
  (∃ f : ℚ → ℚ, (∀ x, f x = x^2 + p*x + q) ∧ f p = 0 ∧ f q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l355_35553


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l355_35523

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l355_35523


namespace NUMINAMATH_CALUDE_g_at_3_l355_35580

def g (x : ℝ) : ℝ := 5 * x^3 - 3 * x^2 + 7 * x - 2

theorem g_at_3 : g 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_g_at_3_l355_35580


namespace NUMINAMATH_CALUDE_jack_emails_l355_35531

theorem jack_emails (morning_emails : ℕ) (difference : ℕ) (afternoon_emails : ℕ) : 
  morning_emails = 6 → 
  difference = 4 → 
  morning_emails = afternoon_emails + difference → 
  afternoon_emails = 2 := by
sorry

end NUMINAMATH_CALUDE_jack_emails_l355_35531


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l355_35508

theorem fraction_simplest_form (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l355_35508


namespace NUMINAMATH_CALUDE_arrangements_count_l355_35512

/-- The number of possible arrangements for 5 male students and 3 female students
    standing in a row, where the female students must stand together. -/
def num_arrangements : ℕ :=
  let num_male_students : ℕ := 5
  let num_female_students : ℕ := 3
  num_male_students.factorial * (num_male_students + 1) * num_female_students.factorial

theorem arrangements_count : num_arrangements = 720 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l355_35512


namespace NUMINAMATH_CALUDE_expression_simplification_l355_35540

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x^2 - 1) / x / (1 + 1/x) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l355_35540


namespace NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l355_35565

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_10 [1, 1, 2, 2, 1, 1]
  first_digit_base_9 y = 4 := by sorry

end NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l355_35565


namespace NUMINAMATH_CALUDE_equation_system_solution_l355_35526

/-- A system of 1000 equations where each x_i^2 = a * x_{i+1} + 1, with x_1000 wrapping back to x_1 -/
def EquationSystem (a : ℝ) (x : Fin 1000 → ℝ) : Prop :=
  ∀ i : Fin 1000, x i ^ 2 = a * x (i.succ) + 1

/-- The solutions to the equation system -/
def Solutions (a : ℝ) : Set ℝ :=
  {x | x = (a + Real.sqrt (a^2 + 4)) / 2 ∨ x = (a - Real.sqrt (a^2 + 4)) / 2}

theorem equation_system_solution (a : ℝ) (ha : |a| > 1) :
  ∀ x : Fin 1000 → ℝ, EquationSystem a x ↔ (∀ i, x i ∈ Solutions a) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l355_35526


namespace NUMINAMATH_CALUDE_marcella_shoes_l355_35549

/-- Given a number of initial shoe pairs and lost individual shoes, 
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) : ℕ :=
  initial_pairs - lost_shoes

/-- Theorem stating that with 24 initial pairs and 9 lost shoes, 
    the maximum number of complete pairs remaining is 15. -/
theorem marcella_shoes : max_remaining_pairs 24 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_l355_35549


namespace NUMINAMATH_CALUDE_function_symmetry_and_value_l355_35550

/-- Given a function f(x) = 2cos(ωx + φ) + m with ω > 0, 
    if f(π/4 - t) = f(t) for all real t and f(π/8) = -1, 
    then m = -3 or m = 1 -/
theorem function_symmetry_and_value (ω φ m : ℝ) (h_ω : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.cos (ω * x + φ) + m) 
  (h_sym : ∀ t, f (π/4 - t) = f t) (h_val : f (π/8) = -1) : 
  m = -3 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_and_value_l355_35550


namespace NUMINAMATH_CALUDE_g_range_l355_35583

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^3) + Real.arctan ((1 - x^3) / (1 + x^3))

theorem g_range :
  Set.range g = Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_g_range_l355_35583


namespace NUMINAMATH_CALUDE_line_mb_equals_nine_l355_35584

/-- Given a line with equation y = mx + b that intersects the y-axis at y = -3
    and rises 3 units for every 1 unit to the right, prove that mb = 9. -/
theorem line_mb_equals_nine (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  (∀ x : ℝ, m * (x + 1) + b = m * x + b + 3) →  -- Slope condition
  m * b = 9 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_nine_l355_35584


namespace NUMINAMATH_CALUDE_root_value_theorem_l355_35559

theorem root_value_theorem (a : ℝ) : a^2 + 3*a + 2 = 0 → a^2 + 3*a = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l355_35559


namespace NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l355_35575

theorem largest_c_for_negative_five_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, x^2 + 4*x + c = -5) ↔ c ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l355_35575


namespace NUMINAMATH_CALUDE_solve_for_y_l355_35569

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 2) (h2 : x = -5) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l355_35569


namespace NUMINAMATH_CALUDE_degree_to_seconds_one_point_four_five_deg_to_seconds_l355_35539

theorem degree_to_seconds (deg : Real) (min_per_deg : Nat) (sec_per_min : Nat) 
  (h1 : min_per_deg = 60) (h2 : sec_per_min = 60) :
  deg * (min_per_deg * sec_per_min) = deg * 3600 := by
  sorry

theorem one_point_four_five_deg_to_seconds :
  1.45 * 3600 = 5220 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_seconds_one_point_four_five_deg_to_seconds_l355_35539


namespace NUMINAMATH_CALUDE_F_sum_positive_l355_35516

/-- The function f(x) = ax^2 + bx + 1 -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function F(x) defined piecewise based on f(x) -/
noncomputable def F (a b x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

/-- Theorem stating that F(m) + F(n) > 0 under given conditions -/
theorem F_sum_positive (a b m n : ℝ) : 
  f a b (-1) = 0 → 
  (∀ x, f a b x ≥ 0) → 
  m * n < 0 → 
  m + n > 0 → 
  a > 0 → 
  (∀ x, f a b x = f a b (-x)) → 
  F a b m + F a b n > 0 := by
  sorry

end NUMINAMATH_CALUDE_F_sum_positive_l355_35516


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l355_35501

-- Define the function
def f (x : ℝ) : ℝ := -(x - 2) * x

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l355_35501


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l355_35504

theorem scientific_notation_equality : (58000000000 : ℝ) = 5.8 * (10 ^ 10) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l355_35504


namespace NUMINAMATH_CALUDE_sales_tax_satisfies_conditions_l355_35541

/-- The sales tax percentage that satisfies the given conditions -/
def sales_tax_percentage : ℝ :=
  -- Define the sales tax percentage
  -- We don't know its exact value yet
  sorry

/-- The cost of the lunch before tax and tip -/
def lunch_cost : ℝ := 100

/-- The tip percentage -/
def tip_percentage : ℝ := 0.06

/-- The total amount paid -/
def total_paid : ℝ := 110

/-- Theorem stating that the sales tax percentage satisfies the given conditions -/
theorem sales_tax_satisfies_conditions :
  lunch_cost + sales_tax_percentage + 
  tip_percentage * (lunch_cost + sales_tax_percentage) = total_paid :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_satisfies_conditions_l355_35541


namespace NUMINAMATH_CALUDE_square_cutout_l355_35589

theorem square_cutout (N M : ℕ) (h : N^2 - M^2 = 79) : M = N - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_cutout_l355_35589


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l355_35571

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x ∨ y = -(Real.sqrt 5 / 2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(√5/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l355_35571


namespace NUMINAMATH_CALUDE_parallel_condition_l355_35535

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) : 
  (line_parallel_plane m β → plane_parallel α β) ∧ 
  ¬(plane_parallel α β → line_parallel_plane m β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l355_35535


namespace NUMINAMATH_CALUDE_apple_problem_l355_35593

/-- Represents the types of Red Fuji apples --/
inductive AppleType
  | A
  | B

/-- Represents the purchase and selling prices for each apple type --/
def price (t : AppleType) : ℕ × ℕ :=
  match t with
  | AppleType.A => (28, 42)
  | AppleType.B => (22, 34)

/-- Represents the total number of apples purchased in the first batch --/
def totalApples : ℕ := 30

/-- Represents the total cost of the first batch of apples --/
def totalCost : ℕ := 720

/-- Represents the maximum number of apples to be purchased in the second batch --/
def maxApples : ℕ := 80

/-- Represents the maximum cost allowed for the second batch --/
def maxCost : ℕ := 2000

/-- Represents the initial daily sales of type B apples at original price --/
def initialSales : ℕ := 4

/-- Represents the increase in daily sales for every 1 yuan price reduction --/
def salesIncrease : ℕ := 2

/-- Represents the target daily profit for type B apples --/
def targetProfit : ℕ := 90

theorem apple_problem :
  ∃ (x y : ℕ),
    x + y = totalApples ∧
    x * (price AppleType.A).1 + y * (price AppleType.B).1 = totalCost ∧
    ∃ (m : ℕ),
      m ≤ maxApples ∧
      m * (price AppleType.A).1 + (maxApples - m) * (price AppleType.B).1 ≤ maxCost ∧
      ∀ (k : ℕ),
        k ≤ maxApples →
        k * (price AppleType.A).1 + (maxApples - k) * (price AppleType.B).1 ≤ maxCost →
        m * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - m) * ((price AppleType.B).2 - (price AppleType.B).1) ≥
        k * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - k) * ((price AppleType.B).2 - (price AppleType.B).1) ∧
    ∃ (a : ℕ),
      (initialSales + salesIncrease * a) * ((price AppleType.B).2 - a - (price AppleType.B).1) = targetProfit :=
by
  sorry


end NUMINAMATH_CALUDE_apple_problem_l355_35593


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l355_35500

/-- Proves that given the conditions of Tammy's mountain climb, her speed on the second day was 4 km/h -/
theorem tammy_mountain_climb (total_time : ℝ) (total_distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_increase = 0.5)
  (h3 : time_decrease = 2)
  (h4 : total_distance = 52) :
  ∃ (speed1 : ℝ) (time1 : ℝ),
    speed1 > 0 ∧
    time1 > 0 ∧
    time1 + (time1 - time_decrease) = total_time ∧
    speed1 * time1 + (speed1 + speed_increase) * (time1 - time_decrease) = total_distance ∧
    speed1 + speed_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l355_35500


namespace NUMINAMATH_CALUDE_ratio_transformation_l355_35513

theorem ratio_transformation (y : ℚ) : 
  (2 + y : ℚ) / (3 + y) = 4 / 5 → (2 + y = 4 ∧ 3 + y = 5) := by
  sorry

#check ratio_transformation

end NUMINAMATH_CALUDE_ratio_transformation_l355_35513


namespace NUMINAMATH_CALUDE_eighteenth_decimal_is_nine_l355_35542

/-- Represents the decimal expansion of a fraction -/
def DecimalExpansion := ℕ → Fin 10

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : DecimalExpansion :=
  fun n => if n % 2 = 0 then 0 else 9

theorem eighteenth_decimal_is_nine
  (h : ∀ n : ℕ, decimal_expansion_10_11 (20 - n) = 9 → decimal_expansion_10_11 n = 9) :
  decimal_expansion_10_11 18 = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_eighteenth_decimal_is_nine_l355_35542


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l355_35515

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range (bounces + 1)) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem: The total distance traveled by a ball dropped from 25 meters,
    rebounding to 2/3 of its previous height for four bounces, is 1900/27 meters -/
theorem ball_bounce_distance :
  totalDistance 25 (2/3) 4 = 1900/27 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l355_35515


namespace NUMINAMATH_CALUDE_square_of_104_l355_35566

theorem square_of_104 : (104 : ℕ)^2 = 10816 := by sorry

end NUMINAMATH_CALUDE_square_of_104_l355_35566


namespace NUMINAMATH_CALUDE_frisbee_cost_l355_35563

/-- The cost of a frisbee given initial money, kite cost, and remaining money --/
theorem frisbee_cost (initial_money kite_cost remaining_money : ℕ) : 
  initial_money = 78 → 
  kite_cost = 8 → 
  remaining_money = 61 → 
  initial_money - kite_cost - remaining_money = 9 := by
sorry

end NUMINAMATH_CALUDE_frisbee_cost_l355_35563


namespace NUMINAMATH_CALUDE_cos_thirteen_pi_fourths_l355_35574

theorem cos_thirteen_pi_fourths : Real.cos (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirteen_pi_fourths_l355_35574


namespace NUMINAMATH_CALUDE_same_color_probability_is_two_twentyfifths_l355_35502

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (silver : ℕ)
  (total : ℕ)
  (h_total : purple + green + blue + silver = total)

/-- The probability of getting the same color on all three dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple : ℚ) ^ 3 / d.total ^ 3 +
  (d.green : ℚ) ^ 3 / d.total ^ 3 +
  (d.blue : ℚ) ^ 3 / d.total ^ 3 +
  (d.silver : ℚ) ^ 3 / d.total ^ 3

/-- The specific die configuration in the problem -/
def problem_die : ColoredDie :=
  { purple := 6
  , green := 8
  , blue := 10
  , silver := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of getting the same color on all three dice -/
theorem same_color_probability_is_two_twentyfifths :
  same_color_probability problem_die = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_two_twentyfifths_l355_35502


namespace NUMINAMATH_CALUDE_university_packaging_cost_l355_35570

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the pricing scheme for boxes -/
structure BoxPricing where
  initialPrice : ℝ
  initialQuantity : ℕ
  additionalPrice : ℝ

/-- Calculates the minimum cost for packaging a given number of items -/
def minimumPackagingCost (boxDim : BoxDimensions) (pricing : BoxPricing) (itemCount : ℕ) : ℝ :=
  let initialCost := pricing.initialPrice * pricing.initialQuantity
  let additionalBoxes := max (itemCount - pricing.initialQuantity) 0
  let additionalCost := pricing.additionalPrice * additionalBoxes
  initialCost + additionalCost

/-- Theorem stating the minimum packaging cost for the university's collection -/
theorem university_packaging_cost :
  let boxDim : BoxDimensions := { length := 18, width := 22, height := 15 }
  let pricing : BoxPricing := { initialPrice := 0.60, initialQuantity := 100, additionalPrice := 0.55 }
  let itemCount : ℕ := 127
  minimumPackagingCost boxDim pricing itemCount = 74.85 := by
  sorry


end NUMINAMATH_CALUDE_university_packaging_cost_l355_35570


namespace NUMINAMATH_CALUDE_two_props_true_l355_35545

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 5 → x^2 - 8*x + 15 = 0

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 - 8*x + 15 = 0 → x = 5

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 5 → x^2 - 8*x + 15 ≠ 0

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 - 8*x + 15 ≠ 0 → x ≠ 5

-- Theorem stating that exactly two propositions (including the original) are true
theorem two_props_true :
  (∀ x, original_prop x) ∧
  (¬ ∀ x, converse_prop x) ∧
  (¬ ∀ x, inverse_prop x) ∧
  (∀ x, contrapositive_prop x) :=
sorry

end NUMINAMATH_CALUDE_two_props_true_l355_35545


namespace NUMINAMATH_CALUDE_max_positive_terms_is_seven_l355_35598

/-- An arithmetic sequence with a positive first term where the sum of the first 3 terms 
    equals the sum of the first 11 terms. -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  first_term_positive : 0 < a₁
  sum_equality : 3 * (2 * a₁ + 2 * d) = 11 * (2 * a₁ + 10 * d)

/-- The maximum number of terms that can be summed before reaching a non-positive term -/
def max_positive_terms (seq : ArithmeticSequence) : ℕ :=
  7

/-- Theorem stating that the maximum number of terms is correct -/
theorem max_positive_terms_is_seven (seq : ArithmeticSequence) :
  (max_positive_terms seq = 7) ∧
  (∀ n : ℕ, n ≤ 7 → seq.a₁ + (n - 1) * seq.d > 0) ∧
  (seq.a₁ + 7 * seq.d ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_positive_terms_is_seven_l355_35598


namespace NUMINAMATH_CALUDE_family_ages_solution_l355_35573

/-- Represents the ages of a family members -/
structure FamilyAges where
  son : ℕ
  daughter : ℕ
  man : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = ages.son + 20 ∧
  ages.man = ages.daughter + 15 ∧
  ages.man + 2 = 2 * (ages.son + 2) ∧
  ages.man + 2 = 3 * (ages.daughter + 2)

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧
    ages.son = 18 ∧ ages.daughter = 23 ∧ ages.man = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l355_35573


namespace NUMINAMATH_CALUDE_sin_cube_identity_l355_35521

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = -(1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l355_35521


namespace NUMINAMATH_CALUDE_medium_boxes_count_l355_35557

def tape_large : ℕ := 4
def tape_medium : ℕ := 2
def tape_small : ℕ := 1
def tape_label : ℕ := 1
def large_boxes : ℕ := 2
def small_boxes : ℕ := 5
def total_tape : ℕ := 44

theorem medium_boxes_count : 
  ∃ (medium_boxes : ℕ), 
    large_boxes * (tape_large + tape_label) + 
    medium_boxes * (tape_medium + tape_label) + 
    small_boxes * (tape_small + tape_label) = total_tape ∧ 
    medium_boxes = 8 := by
sorry

end NUMINAMATH_CALUDE_medium_boxes_count_l355_35557


namespace NUMINAMATH_CALUDE_tom_game_sale_amount_l355_35595

/-- Calculates the amount received from selling a portion of an asset that has increased in value -/
def sellPartOfAppreciatedAsset (initialValue : ℝ) (appreciationFactor : ℝ) (portionSold : ℝ) : ℝ :=
  initialValue * appreciationFactor * portionSold

/-- Proves that Tom sold his games for $240 -/
theorem tom_game_sale_amount : 
  sellPartOfAppreciatedAsset 200 3 0.4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_tom_game_sale_amount_l355_35595


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l355_35506

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 - 5*x + x^2) = 9) ↔ (x = (5 + Real.sqrt 333) / 2 ∨ x = (5 - Real.sqrt 333) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l355_35506


namespace NUMINAMATH_CALUDE_pams_age_l355_35522

/-- Proves that Pam's current age is 5 years, given the conditions of the problem -/
theorem pams_age (pam_age rena_age : ℕ) 
  (h1 : pam_age = rena_age / 2)
  (h2 : rena_age + 10 = (pam_age + 10) + 5) : 
  pam_age = 5 := by sorry

end NUMINAMATH_CALUDE_pams_age_l355_35522


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l355_35534

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degrees :
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degrees_l355_35534


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_929_div_30_l355_35591

theorem least_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n + x) % d = 0 :=
by sorry

theorem least_addend_for_929_div_30 :
  (∃! x : ℕ, x < 30 ∧ (929 + x) % 30 = 0) ∧
  (∀ y : ℕ, y < 30 ∧ (929 + y) % 30 = 0 → y = 1) :=
by sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_929_div_30_l355_35591


namespace NUMINAMATH_CALUDE_juniors_percentage_l355_35585

/-- Represents the composition of students in a high school sample. -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the percentage of a part relative to the total. -/
def percentage (part : ℕ) (total : ℕ) : ℚ :=
  (part : ℚ) / (total : ℚ) * 100

/-- Theorem stating the percentage of juniors in the given student sample. -/
theorem juniors_percentage (sample : StudentSample) : 
  sample.total = 800 ∧ 
  sample.seniors = 160 ∧
  sample.sophomores = sample.total / 4 ∧
  sample.freshmen = sample.sophomores + 24 ∧
  sample.total = sample.freshmen + sample.sophomores + sample.juniors + sample.seniors →
  percentage sample.juniors sample.total = 27 := by
  sorry

end NUMINAMATH_CALUDE_juniors_percentage_l355_35585


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l355_35578

theorem arithmetic_sequence_problem :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence property
  (a 4 + a 5 + a 6 + a 7 = 56) →    -- given condition
  (a 4 * a 7 = 187) →               -- given condition
  ((a 1 = 5 ∧ d = 2) ∨ (a 1 = 23 ∧ d = -2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l355_35578


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l355_35564

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + (4 / 9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - (4 / 9 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l355_35564


namespace NUMINAMATH_CALUDE_max_xy_given_sum_l355_35572

theorem max_xy_given_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_given_sum_l355_35572


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l355_35561

-- Problem 1
theorem problem_1 (a b c : ℚ) : (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = 3/2 * b * c := by sorry

-- Problem 2
theorem problem_2 (m n : ℚ) : (-3*m - 2*n) * (3*m + 2*n) = -9*m^2 - 12*m*n - 4*n^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (h : y ≠ 0) : ((x - 2*y)^2 - (x - 2*y)*(x + 2*y)) / (2*y) = -2*x + 4*y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l355_35561


namespace NUMINAMATH_CALUDE_cake_baking_fraction_l355_35532

theorem cake_baking_fraction (total_cakes : ℕ) 
  (h1 : total_cakes = 60) 
  (initially_baked : ℕ) 
  (h2 : initially_baked = total_cakes / 2) 
  (first_day_baked : ℕ) 
  (h3 : first_day_baked = (total_cakes - initially_baked) / 2) 
  (second_day_baked : ℕ) 
  (h4 : total_cakes - initially_baked - first_day_baked - second_day_baked = 10) :
  (second_day_baked : ℚ) / (total_cakes - initially_baked - first_day_baked) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cake_baking_fraction_l355_35532


namespace NUMINAMATH_CALUDE_mans_to_sons_age_ratio_l355_35562

/-- Given a man who is 28 years older than his son, and the son's present age is 26,
    prove that the ratio of the man's age to his son's age in two years is 2:1. -/
theorem mans_to_sons_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 26 → 
  man_age = son_age + 28 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_to_sons_age_ratio_l355_35562


namespace NUMINAMATH_CALUDE_solve_for_x_l355_35552

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l355_35552


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l355_35558

def total_students : ℕ := 10
def male_students : ℕ := 6
def female_students : ℕ := 4
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l355_35558


namespace NUMINAMATH_CALUDE_function_behavior_l355_35524

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
    (h_odd : is_odd f)
    (h_periodic : ∀ x, f x = f (x - 2))
    (h_decreasing : is_decreasing_on f 1 2) :
  is_decreasing_on f (-3) (-2) ∧ is_increasing_on f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l355_35524


namespace NUMINAMATH_CALUDE_calculate_product_l355_35592

theorem calculate_product : 
  (0.125 : ℝ)^3 * (-8 : ℝ)^3 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l355_35592


namespace NUMINAMATH_CALUDE_min_value_sum_l355_35505

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 2*a*b + 4*b*c + 2*c*a = 16) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + 2*x*y + 4*y*z + 2*z*x = 16 → x + y + z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l355_35505


namespace NUMINAMATH_CALUDE_total_shaded_area_l355_35519

/-- Calculates the total shaded area of a floor tiled with patterned square tiles. -/
theorem total_shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : 
  floor_length = 8 ∧ 
  floor_width = 10 ∧ 
  tile_size = 2 ∧ 
  circle_radius = 1 →
  (floor_length * floor_width / (tile_size * tile_size)) * (tile_size * tile_size - π * circle_radius ^ 2) = 80 - 20 * π :=
by sorry

end NUMINAMATH_CALUDE_total_shaded_area_l355_35519


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l355_35538

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l355_35538


namespace NUMINAMATH_CALUDE_no_valid_N_l355_35576

theorem no_valid_N : ¬ ∃ (N P R : ℕ), 
  N < 40 ∧ 
  P + R = N ∧ 
  (71 * P + 56 * R : ℚ) / N = 66 ∧
  (76 * P : ℚ) / P = 75 ∧
  (61 * R : ℚ) / R = 59 ∧
  P = 2 * R :=
by sorry

end NUMINAMATH_CALUDE_no_valid_N_l355_35576


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l355_35582

theorem ratio_equation_solution (a b c : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 4 * a - c)
  (h3 : c = a + 2) :
  a = 13 / 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l355_35582


namespace NUMINAMATH_CALUDE_integer_sum_difference_product_square_difference_l355_35520

theorem integer_sum_difference_product_square_difference 
  (a b : ℕ+) 
  (sum_eq : a + b = 40)
  (diff_eq : a - b = 8) : 
  a * b = 384 ∧ a^2 - b^2 = 320 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_difference_product_square_difference_l355_35520


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2015_l355_35525

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The cycle length of the last two digits of 8^n -/
def cycleLengthOfLastTwoDigits : ℕ := 20

theorem tens_digit_of_8_pow_2015 :
  ∃ (f : ℕ → ℕ),
    (∀ n, f n = lastTwoDigits n) ∧
    (∀ n, f (n + cycleLengthOfLastTwoDigits) = f n) ∧
    (f 15 = 32) →
    (8^2015 / 10) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2015_l355_35525


namespace NUMINAMATH_CALUDE_factor_expression_l355_35597

theorem factor_expression (x y z : ℝ) :
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) + 
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) + 
  (x^2 - y^2) * (1 + y*z) * (1 + x*z) = 
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l355_35597


namespace NUMINAMATH_CALUDE_acid_dilution_l355_35555

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.40 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l355_35555


namespace NUMINAMATH_CALUDE_train_speed_l355_35551

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1000) (h2 : time = 200) :
  length / time = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l355_35551


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l355_35530

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  (a 4 * a 14 = 8) →
  (∀ x y : ℝ, 2 * a 7 + a 11 ≥ x + y → x * y ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l355_35530


namespace NUMINAMATH_CALUDE_subtracted_number_l355_35537

theorem subtracted_number (x : ℤ) (y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l355_35537


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l355_35518

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l355_35518
