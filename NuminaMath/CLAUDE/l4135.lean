import Mathlib

namespace NUMINAMATH_CALUDE_justin_age_l4135_413529

/-- Prove that Justin's age is 26 years -/
theorem justin_age :
  ∀ (justin_age jessica_age james_age : ℕ),
  (jessica_age = justin_age + 6) →
  (james_age = jessica_age + 7) →
  (james_age + 5 = 44) →
  justin_age = 26 := by
sorry

end NUMINAMATH_CALUDE_justin_age_l4135_413529


namespace NUMINAMATH_CALUDE_number_of_team_formations_l4135_413591

def male_athletes : ℕ := 5
def female_athletes : ℕ := 5
def team_size : ℕ := 6
def ma_long_selected : Prop := true
def ding_ning_selected : Prop := true

def remaining_male_athletes : ℕ := male_athletes - 1
def remaining_female_athletes : ℕ := female_athletes - 1
def remaining_slots : ℕ := team_size - 2

theorem number_of_team_formations :
  (Nat.choose remaining_male_athletes (remaining_slots / 2))^2 *
  (Nat.factorial remaining_slots) =
  number_of_ways_to_form_teams :=
sorry

end NUMINAMATH_CALUDE_number_of_team_formations_l4135_413591


namespace NUMINAMATH_CALUDE_brandon_rabbit_catching_l4135_413517

/-- The number of squirrels Brandon can catch in an hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

/-- The number of rabbits Brandon can catch in an hour -/
def rabbits_per_hour : ℕ := 2

theorem brandon_rabbit_catching :
  squirrels_per_hour * calories_per_squirrel =
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end NUMINAMATH_CALUDE_brandon_rabbit_catching_l4135_413517


namespace NUMINAMATH_CALUDE_ratio_and_quadratic_equation_solution_l4135_413548

theorem ratio_and_quadratic_equation_solution (x y z a : ℤ) : 
  (∃ k : ℚ, x = 4 * k ∧ y = 6 * k ∧ z = 10 * k) →
  y^2 = 40 * a - 20 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_quadratic_equation_solution_l4135_413548


namespace NUMINAMATH_CALUDE_cindy_envelopes_l4135_413581

def envelopes_problem (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : Prop :=
  initial_envelopes - (num_friends * envelopes_per_friend) = 22

theorem cindy_envelopes : envelopes_problem 37 5 3 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l4135_413581


namespace NUMINAMATH_CALUDE_circle_tangency_l4135_413549

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (0, 0) (2, 4) (Real.sqrt 5) (Real.sqrt (20 + m)) → m = -15 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l4135_413549


namespace NUMINAMATH_CALUDE_joeys_route_length_l4135_413502

theorem joeys_route_length 
  (time_one_way : ℝ) 
  (avg_speed : ℝ) 
  (return_speed : ℝ) 
  (h1 : time_one_way = 1)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 12) : 
  ∃ (route_length : ℝ), route_length = 6 ∧ 
    route_length / return_speed + time_one_way = 2 * route_length / avg_speed :=
by sorry

end NUMINAMATH_CALUDE_joeys_route_length_l4135_413502


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l4135_413537

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l4135_413537


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l4135_413584

theorem min_value_quadratic_form (x y : ℝ) : x^2 + 3*x*y + y^2 ≥ 0 ∧ 
  (x^2 + 3*x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l4135_413584


namespace NUMINAMATH_CALUDE_smallest_integer_fourth_root_l4135_413533

theorem smallest_integer_fourth_root (p : ℕ) (q : ℕ) (s : ℝ) : 
  (0 < q) → 
  (0 < s) → 
  (s < 1 / 2000) → 
  (p^(1/4 : ℝ) = q + s) → 
  (∀ (p' : ℕ) (q' : ℕ) (s' : ℝ), 
    0 < q' → 0 < s' → s' < 1 / 2000 → p'^(1/4 : ℝ) = q' + s' → p' ≥ p) →
  q = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_fourth_root_l4135_413533


namespace NUMINAMATH_CALUDE_terms_are_like_when_k_is_two_l4135_413505

/-- Two monomials are like terms if they have the same variables raised to the same powers -/
def like_terms (term1 term2 : ℕ → ℕ) : Prop :=
  ∀ var, term1 var = term2 var

/-- The first term: -3x²y³ᵏ -/
def term1 (k : ℕ) : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 3 * k  -- y has power 3k
| _ => 0  -- other variables have power 0

/-- The second term: 4x²y⁶ -/
def term2 : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 6  -- y has power 6
| _ => 0  -- other variables have power 0

/-- Theorem: When k = 2, -3x²y³ᵏ and 4x²y⁶ are like terms -/
theorem terms_are_like_when_k_is_two : like_terms (term1 2) term2 := by
  sorry

end NUMINAMATH_CALUDE_terms_are_like_when_k_is_two_l4135_413505


namespace NUMINAMATH_CALUDE_failed_both_subjects_percentage_l4135_413521

def total_candidates : ℕ := 3000
def failed_english_percent : ℚ := 49 / 100
def failed_hindi_percent : ℚ := 36 / 100
def passed_english_alone : ℕ := 630

theorem failed_both_subjects_percentage :
  let passed_english_alone_percent : ℚ := passed_english_alone / total_candidates
  let passed_english_percent : ℚ := 1 - failed_english_percent
  let passed_hindi_percent : ℚ := 1 - failed_hindi_percent
  let passed_both_percent : ℚ := passed_english_percent - passed_english_alone_percent
  let passed_hindi_alone_percent : ℚ := passed_hindi_percent - passed_both_percent
  let failed_both_percent : ℚ := 1 - (passed_english_alone_percent + passed_hindi_alone_percent + passed_both_percent)
  failed_both_percent = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_failed_both_subjects_percentage_l4135_413521


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l4135_413514

def complex_mul (a b c d : ℝ) : ℂ :=
  (a * c - b * d : ℝ) + (a * d + b * c : ℝ) * Complex.I

theorem imaginary_part_of_product :
  let z₁ : ℂ := 1 - Complex.I
  let z₂ : ℂ := 2 + 4 * Complex.I
  Complex.im (z₁ * z₂) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l4135_413514


namespace NUMINAMATH_CALUDE_bitcoin_transfer_theorem_l4135_413519

/-- Represents the state of bitcoin holdings for the three businessmen -/
structure BitcoinState where
  sasha : ℕ
  pasha : ℕ
  arkasha : ℕ

/-- Performs the series of transfers described in the problem -/
def perform_transfers (initial : BitcoinState) : BitcoinState :=
  let state1 := BitcoinState.mk (initial.sasha - initial.pasha) (2 * initial.pasha) initial.arkasha
  let state2 := BitcoinState.mk (state1.sasha - initial.arkasha) state1.pasha (2 * initial.arkasha)
  let state3 := BitcoinState.mk (2 * state2.sasha) (state2.pasha - state2.sasha - state2.arkasha) (2 * state2.arkasha)
  BitcoinState.mk (state3.sasha + state3.sasha) (state3.pasha + state3.sasha) (state3.arkasha - state3.sasha - state3.pasha)

/-- The theorem stating the initial and final states of bitcoin holdings -/
theorem bitcoin_transfer_theorem (initial : BitcoinState) :
  initial.sasha = 13 ∧ initial.pasha = 7 ∧ initial.arkasha = 4 ↔
  let final := perform_transfers initial
  final.sasha = 8 ∧ final.pasha = 8 ∧ final.arkasha = 8 :=
sorry

end NUMINAMATH_CALUDE_bitcoin_transfer_theorem_l4135_413519


namespace NUMINAMATH_CALUDE_arrangements_proof_l4135_413563

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3
def tasks : ℕ := 3

def arrangements_with_at_least_one_girl : ℕ :=
  Nat.choose total_people selected_people * Nat.factorial tasks -
  Nat.choose boys selected_people * Nat.factorial tasks

theorem arrangements_proof : arrangements_with_at_least_one_girl = 186 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_proof_l4135_413563


namespace NUMINAMATH_CALUDE_solve_system_l4135_413570

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7) 
  (eq2 : x + 3 * y = 8) : 
  x = 37 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l4135_413570


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_values_l4135_413522

-- Define the points A, B, and C in the plane
def A (a : ℝ) : ℝ × ℝ := (1, -a)
def B (a : ℝ) : ℝ × ℝ := (2, a^2)
def C (a : ℝ) : ℝ × ℝ := (3, a^3)

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem collinear_points_imply_a_values (a : ℝ) :
  collinear (A a) (B a) (C a) → a = 0 ∨ a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_values_l4135_413522


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l4135_413598

/-- The cost of fruits at Lola's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 9 apples = 5 oranges

/-- The theorem stating the relationship between bananas and oranges -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h1 : fc.banana_apple_ratio = 4 / 3)
  (h2 : fc.apple_orange_ratio = 9 / 5) : 
  24 * (fc.apple_orange_ratio * fc.banana_apple_ratio) = 10 := by
  sorry

#check banana_orange_equivalence

end NUMINAMATH_CALUDE_banana_orange_equivalence_l4135_413598


namespace NUMINAMATH_CALUDE_frame_width_is_five_l4135_413542

/-- Represents a frame with square openings -/
structure SquareFrame where
  numOpenings : ℕ
  openingPerimeter : ℝ
  totalPerimeter : ℝ

/-- Calculates the width of the frame -/
def frameWidth (frame : SquareFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with 3 square openings, 
    an opening perimeter of 60 cm, and a total perimeter of 180 cm, 
    the frame width is 5 cm -/
theorem frame_width_is_five :
  let frame : SquareFrame := {
    numOpenings := 3,
    openingPerimeter := 60,
    totalPerimeter := 180
  }
  frameWidth frame = 5 := by sorry

end NUMINAMATH_CALUDE_frame_width_is_five_l4135_413542


namespace NUMINAMATH_CALUDE_bolzano_weierstrass_l4135_413510

-- Define a bounded sequence
def BoundedSequence (a : ℕ → ℝ) : Prop :=
  ∃ (M : ℝ), ∀ (n : ℕ), |a n| ≤ M

-- Define a limit point
def LimitPoint (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ (ε : ℝ), ε > 0 → ∀ (N : ℕ), ∃ (n : ℕ), n ≥ N ∧ |a n - x| < ε

-- Bolzano-Weierstrass theorem
theorem bolzano_weierstrass (a : ℕ → ℝ) :
  BoundedSequence a → ∃ (x : ℝ), LimitPoint a x :=
sorry

end NUMINAMATH_CALUDE_bolzano_weierstrass_l4135_413510


namespace NUMINAMATH_CALUDE_probability_jack_or_queen_l4135_413547

theorem probability_jack_or_queen (total_cards : ℕ) (jack_queen_count : ℕ) 
  (h1 : total_cards = 104) 
  (h2 : jack_queen_count = 16) : 
  (jack_queen_count : ℚ) / total_cards = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_jack_or_queen_l4135_413547


namespace NUMINAMATH_CALUDE_polygon_E_largest_area_l4135_413593

-- Define the polygons and their areas
def polygon_A_area : ℝ := 4
def polygon_B_area : ℝ := 4.5
def polygon_C_area : ℝ := 4.5
def polygon_D_area : ℝ := 5
def polygon_E_area : ℝ := 5.5

-- Define a function to compare areas
def has_largest_area (x y z w v : ℝ) : Prop :=
  v ≥ x ∧ v ≥ y ∧ v ≥ z ∧ v ≥ w

-- Theorem statement
theorem polygon_E_largest_area :
  has_largest_area polygon_A_area polygon_B_area polygon_C_area polygon_D_area polygon_E_area :=
sorry

end NUMINAMATH_CALUDE_polygon_E_largest_area_l4135_413593


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l4135_413575

theorem complex_simplification_and_multiplication :
  3 * ((4 - 3*Complex.I) - (2 + 5*Complex.I)) = 6 - 24*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l4135_413575


namespace NUMINAMATH_CALUDE_possible_values_of_x_l4135_413511

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem possible_values_of_x (x : ℝ) :
  A x ∩ B x = B x → x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l4135_413511


namespace NUMINAMATH_CALUDE_base8_52_equals_base10_42_l4135_413552

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem base8_52_equals_base10_42 :
  base8ToBase10 [5, 2] = 42 := by
  sorry

end NUMINAMATH_CALUDE_base8_52_equals_base10_42_l4135_413552


namespace NUMINAMATH_CALUDE_pattern_equality_l4135_413595

theorem pattern_equality (n : ℤ) : n * (n + 2) - (n + 1)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l4135_413595


namespace NUMINAMATH_CALUDE_largest_quantity_l4135_413555

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2012 / 2011 + 2010 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : C > A ∧ C > B := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l4135_413555


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l4135_413569

def i : ℂ := Complex.I

theorem simplify_complex_fraction :
  (4 + 2 * i) / (4 - 2 * i) - (4 - 2 * i) / (4 + 2 * i) = 8 * i / 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l4135_413569


namespace NUMINAMATH_CALUDE_max_product_of_prime_factors_l4135_413506

def primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem max_product_of_prime_factors :
  ∃ (a b c d e f g : Nat),
    a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧
    e ∈ primes ∧ f ∈ primes ∧ g ∈ primes ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    (a + b + c + d) * (e + f + g) = 841 ∧
    ∀ (x y z w u v t : Nat),
      x ∈ primes → y ∈ primes → z ∈ primes → w ∈ primes →
      u ∈ primes → v ∈ primes → t ∈ primes →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧ x ≠ t ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧ y ≠ t ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧ z ≠ t ∧
      w ≠ u ∧ w ≠ v ∧ w ≠ t ∧
      u ≠ v ∧ u ≠ t ∧
      v ≠ t →
      (x + y + z + w) * (u + v + t) ≤ 841 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_prime_factors_l4135_413506


namespace NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l4135_413513

/-- The time it takes for a motorcyclist and a cyclist to meet under specific conditions -/
theorem meeting_time : ℝ → Prop := fun t =>
  ∀ (D vm vb : ℝ),
  D > 0 →  -- Total distance between A and B is positive
  vm > 0 →  -- Motorcyclist's speed is positive
  vb > 0 →  -- Cyclist's speed is positive
  (1/3) * vm = D/2 + 2 →  -- Motorcyclist's position after 20 minutes
  (1/2) * vb = D/2 - 3 →  -- Cyclist's position after 30 minutes
  t * (vm + vb) = D →  -- They meet when they cover the total distance
  t = 24/60  -- The meeting time is 24 minutes (converted to hours)

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : meeting_time (24/60) := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l4135_413513


namespace NUMINAMATH_CALUDE_additional_telephone_lines_l4135_413528

theorem additional_telephone_lines :
  (9 * 10^6 : ℕ) - (9 * 10^5 : ℕ) = 81 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_additional_telephone_lines_l4135_413528


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l4135_413543

open Real

theorem sine_cosine_shift (ω : ℝ) :
  (∀ x, sin (ω * (x + π / 3)) = cos (ω * x)) → ω = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l4135_413543


namespace NUMINAMATH_CALUDE_square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l4135_413564

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

theorem square_plus_divisor_not_perfect (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 2*n^2) :
  ¬ is_perfect_square (n^2 + d) := by sorry

theorem square_plus_divisor_perfect_iff (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 3*n^2) :
  is_perfect_square (n^2 + d) ↔ d = 3*n^2 := by sorry

end NUMINAMATH_CALUDE_square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l4135_413564


namespace NUMINAMATH_CALUDE_hundred_thousand_scientific_notation_l4135_413579

-- Define scientific notation
def scientific_notation (n : ℝ) (x : ℝ) (y : ℤ) : Prop :=
  n = x * (10 : ℝ) ^ y ∧ 1 ≤ x ∧ x < 10

-- Theorem statement
theorem hundred_thousand_scientific_notation :
  scientific_notation 100000 1 5 :=
by sorry

end NUMINAMATH_CALUDE_hundred_thousand_scientific_notation_l4135_413579


namespace NUMINAMATH_CALUDE_eighth_grade_trip_contribution_l4135_413557

theorem eighth_grade_trip_contribution (total : ℕ) (months : ℕ) 
  (h1 : total = 49685) 
  (h2 : months = 5) : 
  ∃ (students : ℕ) (contribution : ℕ), 
    students * contribution * months = total ∧ 
    students = 19 ∧ 
    contribution = 523 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_trip_contribution_l4135_413557


namespace NUMINAMATH_CALUDE_adults_who_ate_correct_l4135_413550

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := 21

/-- The total number of adults -/
def total_adults : ℕ := 55

/-- The total number of children -/
def total_children : ℕ := 70

/-- The number of adults the meal can fully cater for -/
def meal_capacity_adults : ℕ := 70

/-- The number of children the meal can fully cater for -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be catered with the remaining food after some adults eat -/
def remaining_children_capacity : ℕ := 63

theorem adults_who_ate_correct :
  adults_who_ate * meal_capacity_children / meal_capacity_adults +
  remaining_children_capacity = meal_capacity_children :=
by sorry

end NUMINAMATH_CALUDE_adults_who_ate_correct_l4135_413550


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l4135_413597

/-- Represents the speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- Represents the speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- Represents the time traveled in hours -/
def time : ℝ := 5

/-- Represents the total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l4135_413597


namespace NUMINAMATH_CALUDE_circle_symmetry_max_k_l4135_413578

/-- Given a circle C with center (a,b) and radius 2 passing through (0,2),
    and a line 2x-ky-k=0 with respect to which two points on C are symmetric,
    the maximum value of k is 4√5/5 -/
theorem circle_symmetry_max_k :
  ∀ (a b k : ℝ),
  (a^2 + (b-2)^2 = 4) →  -- circle equation passing through (0,2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ((x₁ - a)^2 + (y₁ - b)^2 = 4) ∧  -- point 1 on circle
    ((x₂ - a)^2 + (y₂ - b)^2 = 4) ∧  -- point 2 on circle
    (2*((x₁ + x₂)/2) - k*((y₁ + y₂)/2) - k = 0) ∧  -- midpoint on line
    (2*a - k*b - k = 0)) →  -- line passes through center
  k ≤ 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_max_k_l4135_413578


namespace NUMINAMATH_CALUDE_incorrect_statement_l4135_413503

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (planesPerp : Plane → Plane → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) : 
  ¬(perpendicular m n ∧ perpendicularToPlane m α ∧ parallelToPlane n β → planesPerp α β) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l4135_413503


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l4135_413585

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed_out : ℕ) 
  (rose_handed_out : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed_out = 120)
  (h3 : rose_handed_out = 320) :
  total_flyers - (jack_handed_out + rose_handed_out) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l4135_413585


namespace NUMINAMATH_CALUDE_journey_distance_ratio_l4135_413509

/-- Proves that the ratio of North distance to East distance is 2:1 given the problem conditions --/
theorem journey_distance_ratio :
  let south_distance : ℕ := 40
  let east_distance : ℕ := south_distance + 20
  let total_distance : ℕ := 220
  let north_distance : ℕ := total_distance - south_distance - east_distance
  (north_distance : ℚ) / east_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_ratio_l4135_413509


namespace NUMINAMATH_CALUDE_unique_factorial_difference_divisibility_l4135_413566

theorem unique_factorial_difference_divisibility :
  ∃! (x : ℕ), x > 0 ∧ (Nat.factorial x - Nat.factorial (x - 4)) / 29 = 1 :=
by
  -- The unique value is 8
  use 8
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_factorial_difference_divisibility_l4135_413566


namespace NUMINAMATH_CALUDE_power_mod_nineteen_l4135_413553

theorem power_mod_nineteen : 11^2048 ≡ 16 [MOD 19] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_nineteen_l4135_413553


namespace NUMINAMATH_CALUDE_wang_loss_l4135_413594

/-- Represents the financial transaction in Mr. Wang's store --/
structure Transaction where
  gift_cost : ℕ
  gift_price : ℕ
  payment : ℕ
  change_given : ℕ
  returned_to_neighbor : ℕ

/-- Calculates the loss in the transaction --/
def calculate_loss (t : Transaction) : ℕ :=
  t.change_given + t.gift_cost + t.returned_to_neighbor - t.payment

/-- Theorem stating that Mr. Wang's loss in the given transaction is $97 --/
theorem wang_loss (t : Transaction) 
  (h1 : t.gift_cost = 18)
  (h2 : t.gift_price = 21)
  (h3 : t.payment = 100)
  (h4 : t.change_given = 79)
  (h5 : t.returned_to_neighbor = 100) : 
  calculate_loss t = 97 := by
  sorry

#eval calculate_loss { gift_cost := 18, gift_price := 21, payment := 100, change_given := 79, returned_to_neighbor := 100 }

end NUMINAMATH_CALUDE_wang_loss_l4135_413594


namespace NUMINAMATH_CALUDE_graph_is_pair_of_lines_l4135_413527

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

/-- The graph is a pair of straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ,
    (is_straight_line f ∧ is_straight_line g) ∧
    ∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_lines_l4135_413527


namespace NUMINAMATH_CALUDE_equal_division_of_money_l4135_413582

theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 5.25) (h2 : num_people = 7) :
  total_amount / num_people = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_money_l4135_413582


namespace NUMINAMATH_CALUDE_function_inequality_l4135_413539

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_ineq : ∀ x, x ≠ 1 → (x - 1) * deriv f x < 0)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = f 0.5)
  (h_b : b = f (4/3))
  (h_c : c = f 3) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_function_inequality_l4135_413539


namespace NUMINAMATH_CALUDE_count_distinct_five_digit_numbers_l4135_413532

/-- The number of distinct five-digit numbers that can be formed by selecting 2 digits
    from the set of odd digits {1, 3, 5, 7, 9} and 3 digits from the set of even digits
    {0, 2, 4, 6, 8}. -/
def distinct_five_digit_numbers : ℕ :=
  let odd_digits : Finset ℕ := {1, 3, 5, 7, 9}
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  10560

/-- Theorem stating that the number of distinct five-digit numbers formed under the given
    conditions is equal to 10560. -/
theorem count_distinct_five_digit_numbers :
  distinct_five_digit_numbers = 10560 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_five_digit_numbers_l4135_413532


namespace NUMINAMATH_CALUDE_health_risk_factors_l4135_413526

theorem health_risk_factors (total_population : ℕ) 
  (prob_one_factor : ℚ) 
  (prob_two_factors : ℚ) 
  (prob_all_given_AB : ℚ) :
  prob_one_factor = 1/10 →
  prob_two_factors = 14/100 →
  prob_all_given_AB = 1/3 →
  total_population > 0 →
  ∃ (num_no_factors : ℕ) (num_not_A : ℕ),
    (num_no_factors : ℚ) / (num_not_A : ℚ) = 21/55 ∧
    num_no_factors + num_not_A = 76 :=
by sorry

end NUMINAMATH_CALUDE_health_risk_factors_l4135_413526


namespace NUMINAMATH_CALUDE_train_passing_bridge_l4135_413507

/-- Time taken for a train to pass a bridge -/
theorem train_passing_bridge
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 870)
  (h2 : train_speed_kmh = 90)
  (h3 : bridge_length = 370) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 49.6 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l4135_413507


namespace NUMINAMATH_CALUDE_only_C_not_proportional_l4135_413538

-- Define the groups of line segments
def group_A : (ℚ × ℚ × ℚ × ℚ) := (3, 6, 2, 4)
def group_B : (ℚ × ℚ × ℚ × ℚ) := (1, 2, 2, 4)
def group_C : (ℚ × ℚ × ℚ × ℚ) := (4, 6, 5, 10)
def group_D : (ℚ × ℚ × ℚ × ℚ) := (1, 1/2, 1/6, 1/3)

-- Define a function to check if a group is proportional
def is_proportional (group : ℚ × ℚ × ℚ × ℚ) : Prop :=
  let (a, b, c, d) := group
  a / b = c / d

-- Theorem stating that only group C is not proportional
theorem only_C_not_proportional :
  is_proportional group_A ∧
  is_proportional group_B ∧
  ¬is_proportional group_C ∧
  is_proportional group_D :=
by sorry

end NUMINAMATH_CALUDE_only_C_not_proportional_l4135_413538


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l4135_413599

/-- Represents the game setup and elimination process -/
structure DealOrNoDeal where
  totalBoxes : Nat
  highValueBoxes : Nat
  eliminatedBoxes : Nat

/-- Checks if the chance of holding a high-value box is at least 1/2 -/
def hasAtLeastHalfChance (game : DealOrNoDeal) : Prop :=
  let remainingBoxes := game.totalBoxes - game.eliminatedBoxes
  2 * game.highValueBoxes ≥ remainingBoxes

/-- The main theorem to prove -/
theorem deal_or_no_deal_elimination 
  (game : DealOrNoDeal) 
  (h1 : game.totalBoxes = 26)
  (h2 : game.highValueBoxes = 6)
  (h3 : game.eliminatedBoxes = 15) : 
  hasAtLeastHalfChance game :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l4135_413599


namespace NUMINAMATH_CALUDE_polynomial_identities_identity1_identity2_identity3_identity4_l4135_413561

-- Define the polynomial identities
theorem polynomial_identities (a b : ℝ) :
  ((a + b) * (a^2 - a*b + b^2) = a^3 + b^3) ∧
  ((a - b) * (a^2 + a*b + b^2) = a^3 - b^3) ∧
  ((a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3) ∧
  (a^3 - 8 = (a - 2) * (a^2 + 2*a + 4)) :=
by sorry

-- Prove each identity separately
theorem identity1 (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 :=
by sorry

theorem identity2 (a b : ℝ) : (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 :=
by sorry

theorem identity3 (a b : ℝ) : (a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3 :=
by sorry

theorem identity4 (a : ℝ) : a^3 - 8 = (a - 2) * (a^2 + 2*a + 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identities_identity1_identity2_identity3_identity4_l4135_413561


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l4135_413524

theorem square_root_equation_solution :
  ∃ x : ℝ, (Real.sqrt 289 - Real.sqrt x / Real.sqrt 25 = 12) ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l4135_413524


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l4135_413540

theorem fraction_sum_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  x / y + y / x ≥ 2 ∧ (x / y + y / x = 2 ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l4135_413540


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4135_413518

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 12) : 
  x + y = 18/5 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4135_413518


namespace NUMINAMATH_CALUDE_min_orders_is_three_l4135_413559

/-- Represents the shopping problem with given conditions -/
structure ShoppingProblem where
  item_price : ℕ  -- Original price of each item in yuan
  item_count : ℕ  -- Number of items
  discount_rate : ℚ  -- Discount rate (e.g., 0.6 for 60% off)
  additional_discount_threshold : ℕ  -- Threshold for additional discount in yuan
  additional_discount_amount : ℕ  -- Amount of additional discount in yuan

/-- Calculates the total cost after discounts for a given number of orders -/
def total_cost (problem : ShoppingProblem) (num_orders : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 3 is the minimum number of orders that minimizes the total cost -/
theorem min_orders_is_three (problem : ShoppingProblem) 
  (h1 : problem.item_price = 48)
  (h2 : problem.item_count = 42)
  (h3 : problem.discount_rate = 0.6)
  (h4 : problem.additional_discount_threshold = 300)
  (h5 : problem.additional_discount_amount = 100) :
  ∀ n : ℕ, n ≠ 3 → total_cost problem 3 ≤ total_cost problem n :=
sorry

end NUMINAMATH_CALUDE_min_orders_is_three_l4135_413559


namespace NUMINAMATH_CALUDE_three_numbers_sum_l4135_413589

theorem three_numbers_sum (x y z M : ℚ) : 
  x + y + z = 48 ∧ 
  x - 5 = M ∧ 
  y + 9 = M ∧ 
  z / 5 = M → 
  M = 52 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l4135_413589


namespace NUMINAMATH_CALUDE_slope_angle_range_l4135_413525

/-- A line passing through the point (0, -2) and intersecting the unit circle -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (0, -2) -/
  passes_through_point : k * 0 - 2 = -2
  /-- The line intersects the unit circle -/
  intersects_circle : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k * x - 2

/-- The slope angle of a line -/
noncomputable def slope_angle (l : IntersectingLine) : ℝ :=
  Real.arctan l.k

/-- Theorem: The range of the slope angle for lines intersecting the unit circle and passing through (0, -2) -/
theorem slope_angle_range (l : IntersectingLine) :
  π/3 ≤ slope_angle l ∧ slope_angle l ≤ 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l4135_413525


namespace NUMINAMATH_CALUDE_inner_polygon_smaller_perimeter_l4135_413587

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  
/-- Calculate the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry

/-- Check if one polygon is contained within another -/
def is_contained_in (inner outer : ConvexPolygon) : Prop :=
  sorry

/-- Theorem: The perimeter of an inner convex polygon is smaller than that of the outer convex polygon -/
theorem inner_polygon_smaller_perimeter
  (inner outer : ConvexPolygon)
  (h_inner_convex : inner.is_convex = true)
  (h_outer_convex : outer.is_convex = true)
  (h_contained : is_contained_in inner outer) :
  perimeter inner < perimeter outer :=
sorry

end NUMINAMATH_CALUDE_inner_polygon_smaller_perimeter_l4135_413587


namespace NUMINAMATH_CALUDE_original_average_weight_l4135_413568

/-- Given a group of students, prove that the original average weight was 28 kg -/
theorem original_average_weight
  (n : ℕ) -- number of original students
  (x : ℝ) -- original average weight
  (w : ℝ) -- weight of new student
  (y : ℝ) -- new average weight after admitting the new student
  (hn : n = 29)
  (hw : w = 13)
  (hy : y = 27.5)
  (h_new_avg : (n : ℝ) * x + w = (n + 1 : ℝ) * y) :
  x = 28 :=
sorry

end NUMINAMATH_CALUDE_original_average_weight_l4135_413568


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l4135_413531

/-- Given a triangle ABC with sides a, b, c in the ratio 2:3:4, 
    it is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (ratio : ∃ (k : ℝ), k > 0 ∧ a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l4135_413531


namespace NUMINAMATH_CALUDE_loan_b_more_cost_effective_l4135_413577

/-- Calculates the total repayable amount for a loan -/
def totalRepayable (principal : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  principal + principal * interestRate * years

/-- Represents the loan options available to Mike -/
structure LoanOption where
  principal : ℝ
  interestRate : ℝ
  years : ℝ

/-- Theorem stating that Loan B is more cost-effective than Loan A -/
theorem loan_b_more_cost_effective (carPrice savings : ℝ) (loanA loanB : LoanOption) :
  carPrice = 35000 ∧
  savings = 5000 ∧
  loanA.principal = 25000 ∧
  loanA.interestRate = 0.07 ∧
  loanA.years = 5 ∧
  loanB.principal = 20000 ∧
  loanB.interestRate = 0.05 ∧
  loanB.years = 4 →
  totalRepayable loanB.principal loanB.interestRate loanB.years <
  totalRepayable loanA.principal loanA.interestRate loanA.years :=
by sorry

end NUMINAMATH_CALUDE_loan_b_more_cost_effective_l4135_413577


namespace NUMINAMATH_CALUDE_triangle_inequality_l4135_413515

/-- A structure representing a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Function to calculate the area of a triangle -/
def areaOfTriangle (t : Triangle) : ℝ := sorry

/-- Function to calculate the tangent of an angle in a triangle -/
def tanAngle (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Function to calculate the perpendicular distance from a point to a line -/
def perpDistance (point : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The main theorem -/
theorem triangle_inequality (t : Triangle) (l : Line) :
  let u := perpDistance t.A l
  let v := perpDistance t.B l
  let w := perpDistance t.C l
  let S := areaOfTriangle t
  u^2 * tanAngle t 0 + v^2 * tanAngle t 1 + w^2 * tanAngle t 2 ≥ 2 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4135_413515


namespace NUMINAMATH_CALUDE_sum_equals_350_l4135_413546

theorem sum_equals_350 : 247 + 53 + 47 + 3 = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_350_l4135_413546


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l4135_413586

/-- The number of ways to arrange 4 volunteers and 1 elder in a row, with the elder in the middle -/
def arrangementCount : ℕ := 24

/-- The number of volunteers -/
def numVolunteers : ℕ := 4

/-- The number of elders -/
def numElders : ℕ := 1

/-- The total number of people -/
def totalPeople : ℕ := numVolunteers + numElders

theorem arrangement_count_is_correct :
  arrangementCount = Nat.factorial numVolunteers := by
  sorry


end NUMINAMATH_CALUDE_arrangement_count_is_correct_l4135_413586


namespace NUMINAMATH_CALUDE_combined_paint_cost_l4135_413556

/-- Represents the dimensions and painting cost of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (breadth : ℝ)
  (paint_rate : ℝ)

/-- Calculates the area of a rectangular floor -/
def floor_area (f : Floor) : ℝ := f.length * f.breadth

/-- Calculates the cost to paint a floor -/
def paint_cost (f : Floor) : ℝ := floor_area f * f.paint_rate

/-- Represents the two-story building -/
structure Building :=
  (first_floor : Floor)
  (second_floor : Floor)

/-- The main theorem to prove -/
theorem combined_paint_cost (b : Building) : ℝ :=
  let f1 := b.first_floor
  let f2 := b.second_floor
  have h1 : f1.length = 3 * f1.breadth := by sorry
  have h2 : paint_cost f1 = 484 := by sorry
  have h3 : f1.paint_rate = 3 := by sorry
  have h4 : f2.length = 0.8 * f1.length := by sorry
  have h5 : f2.breadth = 1.3 * f1.breadth := by sorry
  have h6 : f2.paint_rate = 5 := by sorry
  have h7 : paint_cost f1 + paint_cost f2 = 1320.8 := by sorry
  1320.8

#check combined_paint_cost

end NUMINAMATH_CALUDE_combined_paint_cost_l4135_413556


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l4135_413500

theorem triangle_area_from_squares (a b c : ℝ) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l4135_413500


namespace NUMINAMATH_CALUDE_difference_8_in_96348621_l4135_413560

/-- The difference between the local value and the face value of a digit in a number -/
def localFaceDifference (n : ℕ) (d : ℕ) (p : ℕ) : ℕ :=
  d * (10 ^ p) - d

/-- The position of a digit in a number, counting from right to left and starting at 0 -/
def digitPosition (n : ℕ) (d : ℕ) : ℕ :=
  sorry -- Implementation not required for the statement

theorem difference_8_in_96348621 :
  localFaceDifference 96348621 8 (digitPosition 96348621 8) = 7992 := by
  sorry

end NUMINAMATH_CALUDE_difference_8_in_96348621_l4135_413560


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l4135_413512

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  small_perimeter : ℝ
  width_half_length : width = length / 2
  divides_into_three : length = 3 * width
  small_rect_perimeter : small_perimeter = 2 * (width + length / 3)
  small_perimeter_value : small_perimeter = 40

/-- The perimeter of a SpecialRectangle is 72 -/
theorem special_rectangle_perimeter (rect : SpecialRectangle) : 
  2 * (rect.length + rect.width) = 72 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l4135_413512


namespace NUMINAMATH_CALUDE_next_multiple_age_sum_digits_l4135_413565

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents the family with Joey, Chloe, and Zoe -/
structure Family where
  joey : Person
  chloe : Person
  zoe : Person

/-- Returns true if n is a multiple of m -/
def isMultiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem next_multiple_age_sum_digits (f : Family) : 
  f.zoe.age = 1 →
  f.joey.age = f.chloe.age + 1 →
  (∃ n : ℕ, n ≥ 1 ∧ n ≤ 9 ∧ isMultiple (f.chloe.age + n - 1) n) →
  (∀ m : ℕ, m < f.chloe.age - 1 → ¬isMultiple (f.chloe.age + m - 1) m) →
  (∃ k : ℕ, isMultiple (f.joey.age + k) (f.zoe.age + k) ∧ 
    (∀ j : ℕ, j < k → ¬isMultiple (f.joey.age + j) (f.zoe.age + j)) ∧
    sumOfDigits (f.joey.age + k) = 12) :=
sorry

end NUMINAMATH_CALUDE_next_multiple_age_sum_digits_l4135_413565


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l4135_413554

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

theorem probability_of_letter_in_mathematics :
  (mathematics.toList.toFinset.card : ℚ) / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l4135_413554


namespace NUMINAMATH_CALUDE_calculate_y_investment_y_investment_proof_l4135_413571

/-- Calculates the investment amount of partner y in a business partnership --/
theorem calculate_y_investment (x_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) : ℕ :=
  let y_profit_share := total_profit - x_profit_share
  let y_investment := (y_profit_share * x_investment) / x_profit_share
  y_investment

/-- Proves that y's investment is 15000 given the problem conditions --/
theorem y_investment_proof :
  calculate_y_investment 5000 1600 400 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_y_investment_y_investment_proof_l4135_413571


namespace NUMINAMATH_CALUDE_max_value_of_f_l4135_413562

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧  -- Minimum value is -2
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →  -- Minimum value is achieved
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) ∧   -- Maximum value is at most 1
  (∃ x ∈ Set.Icc 0 1, f a x = 1)     -- Maximum value 1 is achieved
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4135_413562


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4135_413541

/-- Given a hyperbola with equation (x^2 / 9) - (y^2 / 16) = 1, 
    prove its eccentricity and asymptote equations -/
theorem hyperbola_properties :
  let hyperbola := fun (x y : ℝ) => (x^2 / 9) - (y^2 / 16) = 1
  let eccentricity := 5/3
  let asymptote := fun (x : ℝ) => (4/3) * x
  (∀ x y, hyperbola x y → 
    (∃ c, c^2 = 25 ∧ eccentricity = c / 3)) ∧
  (∀ x, hyperbola x (asymptote x) ∨ hyperbola x (-asymptote x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4135_413541


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l4135_413508

/-- Calculates the total number of grandchildren for a grandmother with the given family structure -/
def total_grandchildren (num_daughters num_sons sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  (num_daughters * sons_per_daughter) + (num_sons * daughters_per_son)

/-- Theorem: Grandma Olga's total number of grandchildren is 33 -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

#eval total_grandchildren 3 3 6 5

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l4135_413508


namespace NUMINAMATH_CALUDE_abc_system_solution_l4135_413501

theorem abc_system_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 3 * (a + b))
  (hbc : b * c = 4 * (b + c))
  (hac : a * c = 5 * (a + c)) :
  a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_abc_system_solution_l4135_413501


namespace NUMINAMATH_CALUDE_inverse_power_of_two_l4135_413573

theorem inverse_power_of_two : 2⁻¹ = (1 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_inverse_power_of_two_l4135_413573


namespace NUMINAMATH_CALUDE_logical_equivalence_l4135_413534

variable (E W : Prop)

-- E: Pink elephants on planet α have purple eyes
-- W: Wild boars on planet β have long noses

theorem logical_equivalence :
  ((E → ¬W) ↔ (W → ¬E)) ∧ ((E → ¬W) ↔ (¬E ∨ ¬W)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l4135_413534


namespace NUMINAMATH_CALUDE_glass_volume_l4135_413523

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
sorry

end NUMINAMATH_CALUDE_glass_volume_l4135_413523


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solutions_equation_three_solution_l4135_413545

-- Equation 1
theorem equation_one_solution (x : ℝ) :
  (x^2 + 2) * |2*x - 5| = 0 ↔ x = 5/2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^3 * x = 0 ↔ x = 0 ∨ x = 3 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  |x^4 + 1| = x^4 + x ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solutions_equation_three_solution_l4135_413545


namespace NUMINAMATH_CALUDE_percentage_green_tiles_l4135_413580

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 10
def tiles_per_sqft : ℝ := 4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5
def total_cost : ℝ := 2100

theorem percentage_green_tiles :
  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := (total_cost - red_tile_cost * total_tiles) / (green_tile_cost - red_tile_cost)
  (green_tiles / total_tiles) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_green_tiles_l4135_413580


namespace NUMINAMATH_CALUDE_minji_water_intake_l4135_413558

theorem minji_water_intake (morning_intake : Real) (afternoon_intake : Real)
  (h1 : morning_intake = 0.26)
  (h2 : afternoon_intake = 0.37) :
  morning_intake + afternoon_intake = 0.63 := by
sorry

end NUMINAMATH_CALUDE_minji_water_intake_l4135_413558


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l4135_413592

/-- The number of bottle caps Evelyn starts with -/
def initial_caps : ℕ := 18

/-- The number of bottle caps Evelyn finds -/
def found_caps : ℕ := 63

/-- The total number of bottle caps Evelyn ends up with -/
def total_caps : ℕ := initial_caps + found_caps

theorem evelyn_bottle_caps : total_caps = 81 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l4135_413592


namespace NUMINAMATH_CALUDE_division_problem_l4135_413535

theorem division_problem (x : ℝ) : 45 / x = 900 → x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4135_413535


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l4135_413520

theorem largest_prime_divisor_of_factorial_sum (n : ℕ) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l4135_413520


namespace NUMINAMATH_CALUDE_additional_money_needed_l4135_413590

/-- The amount of money Michael has initially -/
def michaels_money : ℕ := 50

/-- The cost of the cake -/
def cake_cost : ℕ := 20

/-- The cost of the bouquet -/
def bouquet_cost : ℕ := 36

/-- The cost of the balloons -/
def balloon_cost : ℕ := 5

/-- The total cost of all items -/
def total_cost : ℕ := cake_cost + bouquet_cost + balloon_cost

/-- The theorem stating how much more money Michael needs -/
theorem additional_money_needed : total_cost - michaels_money = 11 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l4135_413590


namespace NUMINAMATH_CALUDE_solve_bag_problem_l4135_413551

def bag_problem (total_balls : ℕ) (prob_two_red : ℚ) (red_balls : ℕ) : Prop :=
  total_balls = 10 ∧
  prob_two_red = 1 / 15 ∧
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = prob_two_red

theorem solve_bag_problem :
  ∃ (red_balls : ℕ), bag_problem 10 (1 / 15) red_balls ∧ red_balls = 3 :=
sorry

end NUMINAMATH_CALUDE_solve_bag_problem_l4135_413551


namespace NUMINAMATH_CALUDE_twins_age_problem_l4135_413544

theorem twins_age_problem (age : ℕ) : 
  (age * age) + 5 = ((age + 1) * (age + 1)) → age = 2 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l4135_413544


namespace NUMINAMATH_CALUDE_root_equation_implies_b_equals_four_l4135_413583

theorem root_equation_implies_b_equals_four
  (a b c : ℕ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^3 * (N^2 * N^(1/c))^(1/b))^(1/a) = N^(39/48)) :
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_root_equation_implies_b_equals_four_l4135_413583


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4135_413504

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4135_413504


namespace NUMINAMATH_CALUDE_equation_solutions_l4135_413567

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 905 ∧
    x₂ = -15 + 4 * Real.sqrt 14 ∧
    4 * Real.sqrt 14 > 15 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4135_413567


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l4135_413530

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_only : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_only = 2)
  (h5 : cp.cooking_and_yoga = 7)
  (h6 : cp.all_curriculums = 3) :
  cp.cooking - cp.cooking_only - (cp.cooking_and_yoga - cp.all_curriculums) = 9 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_weaving_count_l4135_413530


namespace NUMINAMATH_CALUDE_optimal_transport_solution_l4135_413536

/-- Represents a vehicle type with its carrying capacity and freight cost. -/
structure VehicleType where
  capacity : ℕ
  cost : ℕ

/-- Represents the transportation problem. -/
structure TransportProblem where
  totalVegetables : ℕ
  totalVehicles : ℕ
  vehicleA : VehicleType
  vehicleB : VehicleType
  vehicleC : VehicleType

/-- Represents a solution to the transportation problem. -/
structure TransportSolution where
  numA : ℕ
  numB : ℕ
  numC : ℕ
  totalCost : ℕ

/-- Checks if a solution is valid for a given problem. -/
def isValidSolution (problem : TransportProblem) (solution : TransportSolution) : Prop :=
  solution.numA + solution.numB + solution.numC = problem.totalVehicles ∧
  solution.numA * problem.vehicleA.capacity +
  solution.numB * problem.vehicleB.capacity +
  solution.numC * problem.vehicleC.capacity ≥ problem.totalVegetables ∧
  solution.totalCost = solution.numA * problem.vehicleA.cost +
                       solution.numB * problem.vehicleB.cost +
                       solution.numC * problem.vehicleC.cost

/-- Theorem stating the optimal solution for the given problem. -/
theorem optimal_transport_solution (problem : TransportProblem)
  (h1 : problem.totalVegetables = 240)
  (h2 : problem.totalVehicles = 16)
  (h3 : problem.vehicleA = ⟨10, 800⟩)
  (h4 : problem.vehicleB = ⟨16, 1000⟩)
  (h5 : problem.vehicleC = ⟨20, 1200⟩) :
  ∃ (solution : TransportSolution),
    isValidSolution problem solution ∧
    solution.numA = 4 ∧
    solution.numB = 10 ∧
    solution.numC = 2 ∧
    solution.totalCost = 15600 ∧
    (∀ (otherSolution : TransportSolution),
      isValidSolution problem otherSolution →
      otherSolution.totalCost ≥ solution.totalCost) :=
sorry

end NUMINAMATH_CALUDE_optimal_transport_solution_l4135_413536


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l4135_413574

theorem polynomial_root_implies_k_value : 
  ∀ k : ℚ, (3 : ℚ)^3 + 7*(3 : ℚ)^2 + k*(3 : ℚ) + 23 = 0 → k = -113/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l4135_413574


namespace NUMINAMATH_CALUDE_weight_2019_is_9_5_l4135_413516

/-- The weight of a stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks in each digit -/
def sticks_in_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_in_digit 2 + sticks_in_digit 0 + sticks_in_digit 1 + sticks_in_digit 9) * stick_weight

/-- Theorem: The weight of the number 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_2019_is_9_5_l4135_413516


namespace NUMINAMATH_CALUDE_line_parallel_properties_l4135_413588

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.angle = l2.angle

-- Theorem statement
theorem line_parallel_properties (l1 l2 : Line) :
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (l1.angle = l2.angle → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle = l2.angle) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_properties_l4135_413588


namespace NUMINAMATH_CALUDE_no_solution_inequality_l4135_413576

theorem no_solution_inequality :
  ¬∃ (x : ℝ), (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l4135_413576


namespace NUMINAMATH_CALUDE_train_average_speed_with_stoppages_l4135_413596

theorem train_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ)
  (stop_time_per_hour : ℝ)
  (h1 : speed_without_stoppages = 100)
  (h2 : stop_time_per_hour = 3)
  : (speed_without_stoppages * (60 - stop_time_per_hour) / 60) = 95 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_with_stoppages_l4135_413596


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l4135_413572

/-- The line y = a is tangent to the circle x^2 + y^2 - 2y = 0 if and only if a = 0 or a = 2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = a → x^2 + y^2 - 2*y = 0 → (x = 0 ∧ (y = a + 1 ∨ y = a - 1))) ↔ (a = 0 ∨ a = 2) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l4135_413572
