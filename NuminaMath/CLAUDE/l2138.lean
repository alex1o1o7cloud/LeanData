import Mathlib

namespace NUMINAMATH_CALUDE_max_sin_a_value_l2138_213864

open Real

theorem max_sin_a_value (a b c : ℝ) 
  (h1 : cos a = tan b) 
  (h2 : cos b = tan c) 
  (h3 : cos c = tan a) : 
  ∃ (max_sin_a : ℝ), (∀ a' b' c' : ℝ, cos a' = tan b' → cos b' = tan c' → cos c' = tan a' → sin a' ≤ max_sin_a) ∧ max_sin_a = (sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_value_l2138_213864


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2138_213862

theorem trigonometric_identity : 
  100 * (Real.sin (253 * π / 180) * Real.sin (313 * π / 180) + 
         Real.sin (163 * π / 180) * Real.sin (223 * π / 180)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2138_213862


namespace NUMINAMATH_CALUDE_factorization_equality_l2138_213825

theorem factorization_equality (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2138_213825


namespace NUMINAMATH_CALUDE_circular_board_area_l2138_213821

/-- The area of a circular wooden board that rolls forward for 10 revolutions
    and advances exactly 62.8 meters is π square meters. -/
theorem circular_board_area (revolutions : ℕ) (distance : ℝ) (area : ℝ) :
  revolutions = 10 →
  distance = 62.8 →
  area = π →
  area = (distance / (2 * revolutions : ℝ))^2 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_board_area_l2138_213821


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l2138_213810

/-- 
Given a circle where an arc of length 5π cm corresponds to a central angle of 150°, 
the radius of the circle is 6 cm.
-/
theorem circle_radius_from_arc_and_angle : 
  ∀ (r : ℝ), 
  (150 / 180 : ℝ) * Real.pi * r = 5 * Real.pi → 
  r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l2138_213810


namespace NUMINAMATH_CALUDE_probability_two_even_balls_l2138_213858

theorem probability_two_even_balls (n : ℕ) (h1 : n = 16) :
  let total_balls := n
  let even_balls := n / 2
  let prob_first_even := even_balls / total_balls
  let prob_second_even := (even_balls - 1) / (total_balls - 1)
  prob_first_even * prob_second_even = 7 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_balls_l2138_213858


namespace NUMINAMATH_CALUDE_debate_team_girls_l2138_213853

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (members_per_group : ℕ) : 
  boys = 26 → groups = 8 → members_per_group = 9 → 
  (groups * members_per_group) - boys = 46 := by sorry

end NUMINAMATH_CALUDE_debate_team_girls_l2138_213853


namespace NUMINAMATH_CALUDE_roof_difference_l2138_213815

theorem roof_difference (width : ℝ) (length : ℝ) (area : ℝ) : 
  width > 0 →
  length = 4 * width →
  area = 588 →
  length * width = area →
  length - width = 21 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_roof_difference_l2138_213815


namespace NUMINAMATH_CALUDE_katie_marbles_l2138_213835

theorem katie_marbles (pink : ℕ) (orange : ℕ) (purple : ℕ) 
  (h1 : pink = 13)
  (h2 : orange = pink - 9)
  (h3 : purple = 4 * orange) :
  pink + orange + purple = 33 := by
sorry

end NUMINAMATH_CALUDE_katie_marbles_l2138_213835


namespace NUMINAMATH_CALUDE_no_divisibility_by_15_and_11_exists_divisibility_by_11_l2138_213885

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def construct_number (n : ℕ) : ℕ :=
  80000 + n * 1000 + 642

theorem no_divisibility_by_15_and_11 :
  ¬ ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 15 = 0 ∧ 
    (construct_number n) % 11 = 0 :=
sorry

theorem exists_divisibility_by_11 :
  ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_divisibility_by_15_and_11_exists_divisibility_by_11_l2138_213885


namespace NUMINAMATH_CALUDE_bean_in_circle_probability_l2138_213890

/-- The probability of a randomly thrown bean landing inside the inscribed circle of an equilateral triangle with side length 2 -/
theorem bean_in_circle_probability : 
  let triangle_side : ℝ := 2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  let circle_radius : ℝ := (Real.sqrt 3 / 3) * triangle_side
  let circle_area : ℝ := Real.pi * circle_radius^2
  let probability : ℝ := circle_area / triangle_area
  probability = (Real.sqrt 3 * Real.pi) / 9 := by
sorry

end NUMINAMATH_CALUDE_bean_in_circle_probability_l2138_213890


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2138_213869

/-- Atomic weight of Copper in g/mol -/
def copper_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def copper_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  copper_count * copper_weight + 
  carbon_count * carbon_weight + 
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 123.554 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight = 123.554 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2138_213869


namespace NUMINAMATH_CALUDE_f_exp_negative_range_l2138_213893

open Real

theorem f_exp_negative_range (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x => x - 1 - (e - 1) * log x
  ∀ x : ℝ, f (exp x) < 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_exp_negative_range_l2138_213893


namespace NUMINAMATH_CALUDE_even_square_iff_even_l2138_213843

theorem even_square_iff_even (p : ℕ) : Even p ↔ Even (p^2) := by
  sorry

end NUMINAMATH_CALUDE_even_square_iff_even_l2138_213843


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2138_213817

/-- The number of different kinds of lunch meats -/
def num_meats : ℕ := 12

/-- The number of different kinds of cheeses -/
def num_cheeses : ℕ := 8

/-- The number of ways to choose meats for a sandwich -/
def meat_choices : ℕ := Nat.choose num_meats 1 + Nat.choose num_meats 2

/-- The number of ways to choose cheeses for a sandwich -/
def cheese_choices : ℕ := Nat.choose num_cheeses 2

/-- The total number of different sandwiches that can be made -/
def total_sandwiches : ℕ := meat_choices * cheese_choices

theorem sandwich_combinations : total_sandwiches = 2184 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2138_213817


namespace NUMINAMATH_CALUDE_system_solution_l2138_213855

theorem system_solution : ∃! (x y z : ℝ), 
  (x - y ≥ z ∧ x^2 + 4*y^2 + 5 = 4*z) ∧ 
  x = 2 ∧ y = -1/2 ∧ z = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2138_213855


namespace NUMINAMATH_CALUDE_cube_volume_increase_l2138_213808

theorem cube_volume_increase (a : ℝ) (ha : a > 0) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l2138_213808


namespace NUMINAMATH_CALUDE_parabola_properties_l2138_213830

/-- Properties of a parabola y = ax^2 + bx + c with a > 0, b > 0, and c < 0 -/
theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  let f := fun x => a * x^2 + b * x + c
  let vertex_x := -b / (2 * a)
  (∀ x y : ℝ, f x < f y → x < y ∨ y < x) ∧  -- Opens upwards
  vertex_x < 0 ∧                            -- Vertex in left half-plane
  f 0 < 0                                   -- Y-intercept below origin
:= by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2138_213830


namespace NUMINAMATH_CALUDE_poster_ratio_l2138_213827

theorem poster_ratio (total medium large small : ℕ) : 
  total = 50 ∧ 
  medium = total / 2 ∧ 
  large = 5 ∧ 
  small = total - medium - large → 
  small * 5 = total * 2 := by
sorry

end NUMINAMATH_CALUDE_poster_ratio_l2138_213827


namespace NUMINAMATH_CALUDE_roots_expression_simplification_l2138_213837

theorem roots_expression_simplification (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0) 
  (h3 : γ^2 + q*γ + 2 = 0) 
  (h4 : δ^2 + q*δ + 2 = 0) : 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 2*(p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_simplification_l2138_213837


namespace NUMINAMATH_CALUDE_second_integer_value_l2138_213861

theorem second_integer_value (n : ℝ) : 
  (n + (n + 3) = 150) → (n + 1 = 74.5) := by
  sorry

end NUMINAMATH_CALUDE_second_integer_value_l2138_213861


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l2138_213846

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, 4; -2, -3] →
  (B^3)⁻¹ = !![3, 4; -2, -3] :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l2138_213846


namespace NUMINAMATH_CALUDE_visibility_time_proof_l2138_213868

/-- Alice's walking speed in feet per second -/
def alice_speed : ℝ := 2

/-- Bob's walking speed in feet per second -/
def bob_speed : ℝ := 4

/-- Distance between Alice and Bob's parallel paths in feet -/
def path_distance : ℝ := 300

/-- Diameter of the circular monument in feet -/
def monument_diameter : ℝ := 150

/-- Initial distance between Alice and Bob when the monument first blocks their line of sight -/
def initial_distance : ℝ := 300

/-- Time until Alice and Bob can see each other again -/
def visibility_time : ℝ := 48

theorem visibility_time_proof :
  alice_speed = 2 ∧
  bob_speed = 4 ∧
  path_distance = 300 ∧
  monument_diameter = 150 ∧
  initial_distance = 300 →
  visibility_time = 48 := by
  sorry

#check visibility_time_proof

end NUMINAMATH_CALUDE_visibility_time_proof_l2138_213868


namespace NUMINAMATH_CALUDE_curve_properties_l2138_213898

-- Define the curve C
def C (k : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - k) + y^2 / (k - 1) = 1}

-- Define what it means for C to be a circle
def is_circle (k : ℝ) := ∃ r : ℝ, ∀ (x y : ℝ), (x, y) ∈ C k → x^2 + y^2 = r^2

-- Define what it means for C to be an ellipse
def is_ellipse (k : ℝ) := ∃ a b : ℝ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 + y^2 / b^2 = 1

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 - y^2 / b^2 = 1 ∨ y^2 / a^2 - x^2 / b^2 = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) := is_ellipse k ∧ ∃ c : ℝ, c > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → (x + c, y) ∈ C k ∧ (x - c, y) ∈ C k

theorem curve_properties :
  (∃ k : ℝ, is_circle k) ∧
  (∃ k : ℝ, 1 < k ∧ k < 4 ∧ ¬is_ellipse k) ∧
  (∀ k : ℝ, is_hyperbola k → k < 1 ∨ k > 4) ∧
  (∀ k : ℝ, is_ellipse_x_foci k → 1 < k ∧ k < 5/2) :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l2138_213898


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2138_213866

theorem max_value_quadratic :
  ∃ (c : ℝ), c = 3395 / 49 ∧ ∀ (r : ℝ), -7 * r^2 + 50 * r - 20 ≤ c := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2138_213866


namespace NUMINAMATH_CALUDE_proper_fraction_triple_when_cubed_l2138_213823

theorem proper_fraction_triple_when_cubed (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b ↔ a = 2 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_proper_fraction_triple_when_cubed_l2138_213823


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2138_213883

def n₁ : ℕ := 263
def n₂ : ℕ := 935
def n₃ : ℕ := 1383
def r : ℕ := 7
def d : ℕ := 32

theorem greatest_divisor_with_remainder (m : ℕ) :
  (m > d → ¬(n₁ % m = r ∧ n₂ % m = r ∧ n₃ % m = r)) ∧
  (n₁ % d = r ∧ n₂ % d = r ∧ n₃ % d = r) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2138_213883


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_72_l2138_213894

/-- Given a train that crosses a platform and a stationary man, calculate its speed in km/h -/
theorem train_speed (platform_length : ℝ) (platform_time : ℝ) (man_time : ℝ) : ℝ :=
  let train_speed_mps := platform_length / (platform_time - man_time)
  let train_speed_kmph := train_speed_mps * 3.6
  train_speed_kmph

/-- The speed of the train is 72 km/h -/
theorem train_speed_is_72 : 
  train_speed 260 31 18 = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_72_l2138_213894


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l2138_213848

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents a set of parallel lines -/
structure ParallelLines where
  ab : LineSegment
  cd : LineSegment
  ef : LineSegment
  gh : LineSegment

/-- Given conditions for the problem -/
def problem_conditions (lines : ParallelLines) : Prop :=
  lines.ab.length = 300 ∧
  lines.cd.length = 200 ∧
  lines.ef.length = (lines.ab.length + lines.cd.length) / 4 ∧
  lines.gh.length = lines.ef.length - (lines.ef.length - lines.cd.length) / 4

/-- The theorem to be proved -/
theorem parallel_lines_theorem (lines : ParallelLines) 
  (h : problem_conditions lines) : lines.gh.length = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l2138_213848


namespace NUMINAMATH_CALUDE_xiaofang_final_score_l2138_213802

/-- Calculates the final score in a speech contest given the scores and weights for each category -/
def calculate_final_score (speech_content_score : ℝ) (language_expression_score : ℝ) (overall_effect_score : ℝ) 
  (speech_content_weight : ℝ) (language_expression_weight : ℝ) (overall_effect_weight : ℝ) : ℝ :=
  speech_content_score * speech_content_weight + 
  language_expression_score * language_expression_weight + 
  overall_effect_score * overall_effect_weight

/-- Theorem stating that Xiaofang's final score is 90 points -/
theorem xiaofang_final_score : 
  calculate_final_score 85 95 90 0.4 0.4 0.2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_xiaofang_final_score_l2138_213802


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2138_213844

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  crossing_time = 20 →
  train_speed_kmh = 70.2 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
    sorry


end NUMINAMATH_CALUDE_bridge_length_calculation_l2138_213844


namespace NUMINAMATH_CALUDE_sum_equals_5070_l2138_213859

theorem sum_equals_5070 (P : ℕ) : 
  1010 + 1012 + 1014 + 1016 + 1018 = 5100 - P → P = 30 :=
by sorry

end NUMINAMATH_CALUDE_sum_equals_5070_l2138_213859


namespace NUMINAMATH_CALUDE_pi_is_irrational_l2138_213865

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- Axiom: All irrational numbers are infinite non-repeating decimals
axiom irrational_are_infinite_non_repeating : 
  ∀ x : ℝ, is_irrational x → is_infinite_non_repeating_decimal x

-- Given: π is an infinite non-repeating decimal
axiom pi_is_infinite_non_repeating : is_infinite_non_repeating_decimal Real.pi

-- Theorem to prove
theorem pi_is_irrational : is_irrational Real.pi := sorry

end NUMINAMATH_CALUDE_pi_is_irrational_l2138_213865


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2138_213882

/-- Proves the factorization of x^2(x-1)-x+1 -/
theorem factorization_1 (x : ℝ) : x^2 * (x - 1) - x + 1 = (x - 1)^2 * (x + 1) := by
  sorry

/-- Proves the factorization of 3p(x+1)^3y^2+6p(x+1)^2y+3p(x+1) -/
theorem factorization_2 (p x y : ℝ) : 
  3 * p * (x + 1)^3 * y^2 + 6 * p * (x + 1)^2 * y + 3 * p * (x + 1) = 
  3 * p * (x + 1) * (x * y + y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2138_213882


namespace NUMINAMATH_CALUDE_puppies_sold_is_24_l2138_213805

/-- Represents the pet store scenario --/
structure PetStore where
  initial_puppies : ℕ
  puppies_per_cage : ℕ
  cages_used : ℕ

/-- Calculates the number of puppies sold --/
def puppies_sold (store : PetStore) : ℕ :=
  store.initial_puppies - (store.puppies_per_cage * store.cages_used)

/-- Theorem stating that 24 puppies were sold --/
theorem puppies_sold_is_24 :
  ∃ (store : PetStore),
    store.initial_puppies = 56 ∧
    store.puppies_per_cage = 4 ∧
    store.cages_used = 8 ∧
    puppies_sold store = 24 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_is_24_l2138_213805


namespace NUMINAMATH_CALUDE_sum_to_zero_l2138_213839

/-- Given an initial sum of 2b - 1, where one addend is increased by 3b - 8 and another is decreased by -b - 7,
    prove that subtracting 6b - 2 from the third addend makes the total sum zero. -/
theorem sum_to_zero (b : ℝ) : 
  let initial_sum := 2*b - 1
  let increase := 3*b - 8
  let decrease := -b - 7
  let subtraction := 6*b - 2
  initial_sum + increase - decrease - subtraction = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_to_zero_l2138_213839


namespace NUMINAMATH_CALUDE_base_subtraction_l2138_213854

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_subtraction :
  let base9_321 := [3, 2, 1]
  let base6_254 := [2, 5, 4]
  (toBase10 base9_321 9) - (toBase10 base6_254 6) = 156 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l2138_213854


namespace NUMINAMATH_CALUDE_combined_pastures_capacity_l2138_213878

/-- Represents the capacity of a pasture -/
structure Pasture where
  area : ℝ
  cattleCapacity : ℕ
  daysCapacity : ℕ

/-- Calculates the total grass units a pasture can provide -/
def totalGrassUnits (p : Pasture) : ℝ :=
  p.area * (p.cattleCapacity : ℝ) * (p.daysCapacity : ℝ)

/-- Theorem: Combined pastures can feed 250 cattle for 28 days -/
theorem combined_pastures_capacity 
  (pastureA : Pasture)
  (pastureB : Pasture)
  (h1 : pastureA.area = 3)
  (h2 : pastureB.area = 4)
  (h3 : pastureA.cattleCapacity = 90)
  (h4 : pastureA.daysCapacity = 36)
  (h5 : pastureB.cattleCapacity = 160)
  (h6 : pastureB.daysCapacity = 24)
  (h7 : totalGrassUnits pastureA + totalGrassUnits pastureB = 
        (pastureA.area + pastureB.area) * 250 * 28) :
  ∃ (combinedPasture : Pasture), 
    combinedPasture.area = pastureA.area + pastureB.area ∧
    combinedPasture.cattleCapacity = 250 ∧
    combinedPasture.daysCapacity = 28 :=
  sorry

end NUMINAMATH_CALUDE_combined_pastures_capacity_l2138_213878


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_6_l2138_213887

theorem sqrt_2_times_sqrt_6 : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_6_l2138_213887


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l2138_213832

theorem gcd_of_squares_sum : Nat.gcd (168^2 + 301^2 + 502^2) (169^2 + 300^2 + 501^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l2138_213832


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l2138_213841

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- The asymptotes of the hyperbola -/
def asymptotes (slope : ℝ) (y_intercept1 y_intercept2 : ℝ) :=
  (fun x => slope * x + y_intercept1, fun x => -slope * x + y_intercept2)

theorem hyperbola_a_plus_h_value
  (slope : ℝ)
  (y_intercept1 y_intercept2 : ℝ)
  (point_x point_y : ℝ)
  (h : Hyperbola)
  (asym : asymptotes slope y_intercept1 y_intercept2 = 
    (fun x => 3 * x + 4, fun x => -3 * x + 2))
  (point_on_hyperbola : (point_x, point_y) = (1, 8))
  (hyperbola_eq : ∀ x y, 
    (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1 ↔ 
    (fun x y => (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1) x y) :
  h.a + h.h = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l2138_213841


namespace NUMINAMATH_CALUDE_object_height_l2138_213852

/-- The height function of an object thrown upward -/
def h (k : ℝ) (t : ℝ) : ℝ := -k * (t - 3)^2 + 150

/-- The value of k for which the object is at 94 feet after 5 seconds -/
theorem object_height (k : ℝ) : h k 5 = 94 → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_object_height_l2138_213852


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2138_213820

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 2) :
  Complex.abs ((z - 1)^2 * (z + 1)) ≤ 4 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 2 ∧ Complex.abs ((w - 1)^2 * (w + 1)) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2138_213820


namespace NUMINAMATH_CALUDE_function_properties_l2138_213831

-- Define the function f and its derivative
def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

-- Main theorem
theorem function_properties (a b m : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f' is symmetric about x = -1/2
  (f' a b 1 = 0) →                         -- f'(1) = 0
  (a = 3 ∧ b = -12) ∧                      -- Part 1: values of a and b
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧     -- f has exactly three zeros
    f 3 (-12) m x₁ = 0 ∧
    f 3 (-12) m x₂ = 0 ∧
    f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7                          -- Part 2: range of m
  := by sorry


end NUMINAMATH_CALUDE_function_properties_l2138_213831


namespace NUMINAMATH_CALUDE_largest_non_sum_of_multiple_30_and_composite_l2138_213875

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_of_multiple_30_and_composite :
  (∀ n > 211, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_multiple_30_and_composite_l2138_213875


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2138_213899

theorem prime_equation_solution :
  ∀ p q : ℕ, 
    Nat.Prime p → Nat.Prime q →
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
    (p = 17 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2138_213899


namespace NUMINAMATH_CALUDE_special_ellipse_ratio_l2138_213809

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  -- Semi-major axis
  a : ℝ
  -- Semi-minor axis
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Ensure a > b > 0 and c > 0
  h1 : a > b
  h2 : b > 0
  h3 : c > 0
  -- Ellipse equation: a² = b² + c²
  h4 : a^2 = b^2 + c^2
  -- Special condition: |F1B2|² = |OF1| * |B1B2|
  h5 : (a + c)^2 = c * (2 * b)

/-- The ratio of semi-major axis to center-focus distance is 3:2 -/
theorem special_ellipse_ratio (e : SpecialEllipse) : a / c = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_ratio_l2138_213809


namespace NUMINAMATH_CALUDE_most_likely_parent_genotypes_l2138_213851

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Represents the phenotype (observable trait) of a rabbit -/
inductive Phenotype
| Hairy
| Smooth

/-- Function to determine the phenotype from a genotype -/
def phenotypeFromGenotype (g : Genotype) : Phenotype :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => Phenotype.Hairy
  | _, Allele.H => Phenotype.Hairy
  | Allele.S, _ => Phenotype.Smooth
  | _, Allele.S => Phenotype.Smooth
  | Allele.h, Allele.h => Phenotype.Hairy
  | Allele.s, Allele.s => Phenotype.Smooth
  | _, _ => Phenotype.Smooth

/-- The probability of the hairy allele in the population -/
def hairyAlleleProbability : ℝ := 0.1

/-- Theorem stating the most likely genotype combination for parents -/
theorem most_likely_parent_genotypes
  (hairyParent smoothParent : Genotype)
  (allOffspringHairy : ∀ (offspring : Genotype),
    phenotypeFromGenotype offspring = Phenotype.Hairy) :
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.S, Allele.h⟩) ∨
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.h, Allele.S⟩) :=
sorry


end NUMINAMATH_CALUDE_most_likely_parent_genotypes_l2138_213851


namespace NUMINAMATH_CALUDE_odd_z_has_4n_minus_1_divisor_l2138_213895

theorem odd_z_has_4n_minus_1_divisor (x y : ℕ+) (z : ℤ) 
  (hz : z = (4 * x * y : ℤ) / (x + y : ℤ)) 
  (hodd : Odd z) : 
  ∃ (d : ℤ), d ∣ z ∧ ∃ (n : ℕ+), d = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_z_has_4n_minus_1_divisor_l2138_213895


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2138_213886

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate indicating if a quadratic polynomial has roots -/
def has_roots (p : QuadraticPolynomial) : Prop :=
  p.b ^ 2 - 4 * p.a * p.c ≥ 0

/-- Given polynomial with coefficients squared -/
def squared_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 2, p.b ^ 2, p.c ^ 2⟩

/-- Given polynomial with coefficients cubed -/
def cubed_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 3, p.b ^ 3, p.c ^ 3⟩

theorem quadratic_roots_theorem (p : QuadraticPolynomial) 
  (h : has_roots p) : 
  (¬ ∀ p, has_roots p → has_roots (squared_poly p)) ∧ 
  (∀ p, has_roots p → has_roots (cubed_poly p)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2138_213886


namespace NUMINAMATH_CALUDE_quadratic_inequality_existence_l2138_213871

theorem quadratic_inequality_existence (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) ↔ (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_existence_l2138_213871


namespace NUMINAMATH_CALUDE_steiner_symmetrization_preserves_convexity_l2138_213822

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define Steiner symmetrization
def SteinerSymmetrization (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem steiner_symmetrization_preserves_convexity
  (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) :
  ConvexPolygon M →
  ConvexPolygon (SteinerSymmetrization M l) := by
  sorry

end NUMINAMATH_CALUDE_steiner_symmetrization_preserves_convexity_l2138_213822


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l2138_213826

/-- The expected value of flipping a set of coins -/
def expected_value (coin_values : List ℚ) : ℚ :=
  (coin_values.map (· / 2)).sum

/-- Theorem: The expected value of flipping a penny, nickel, dime, quarter, and half-dollar is 45.5 cents -/
theorem coin_flip_expected_value :
  expected_value [1, 5, 10, 25, 50] = 91/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l2138_213826


namespace NUMINAMATH_CALUDE_subcommittee_count_l2138_213872

theorem subcommittee_count (n k : ℕ) (hn : n = 8) (hk : k = 3) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l2138_213872


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2138_213806

theorem complex_fraction_equality : (1 + 2*Complex.I) / (1 - Complex.I)^2 = 1 - (1/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2138_213806


namespace NUMINAMATH_CALUDE_blocks_color_theorem_l2138_213863

theorem blocks_color_theorem (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_color_theorem_l2138_213863


namespace NUMINAMATH_CALUDE_tylenol_consumption_l2138_213889

/-- Calculates the total grams of Tylenol taken given the dosage and duration -/
def totalTylenolGrams (tabletsPer4Hours : ℕ) (mgPerTablet : ℕ) (totalHours : ℕ) : ℚ :=
  let dosesCount := totalHours / 4
  let totalTablets := dosesCount * tabletsPer4Hours
  let totalMg := totalTablets * mgPerTablet
  (totalMg : ℚ) / 1000

/-- Theorem stating that under the given conditions, 3 grams of Tylenol are taken -/
theorem tylenol_consumption : totalTylenolGrams 2 500 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_consumption_l2138_213889


namespace NUMINAMATH_CALUDE_notebook_cost_l2138_213828

def total_spent : ℕ := 74
def ruler_cost : ℕ := 18
def pencil_cost : ℕ := 7
def num_pencils : ℕ := 3

theorem notebook_cost :
  total_spent - (ruler_cost + num_pencils * pencil_cost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2138_213828


namespace NUMINAMATH_CALUDE_power_of_power_three_l2138_213840

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2138_213840


namespace NUMINAMATH_CALUDE_simplify_expression_l2138_213892

theorem simplify_expression (a b : ℝ) : 4*a + 5*b - a - 7*b = 3*a - 2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2138_213892


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2138_213857

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2 - 1) = 0) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ax₁ + 2*y₁ + 6 = 0 ∧ 
    x₂ + a*(a+1)*y₂ + (a^2 - 1) = 0 →
    (x₂ - x₁) * (ax₁ + 2*y₁) + (y₂ - y₁) * (2*x₁ - 2*a*y₁) = 0) →
  a = 0 ∨ a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2138_213857


namespace NUMINAMATH_CALUDE_gcd_problem_l2138_213845

theorem gcd_problem : Int.gcd (123^2 + 235^2 - 347^2) (122^2 + 234^2 - 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2138_213845


namespace NUMINAMATH_CALUDE_distance_to_origin_l2138_213850

theorem distance_to_origin (a : ℝ) : |a| = 3 → (a - 2 = 1 ∨ a - 2 = -5) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2138_213850


namespace NUMINAMATH_CALUDE_no_sum_of_squares_representation_l2138_213880

theorem no_sum_of_squares_representation : ¬∃ (n : ℕ), ∃ (x y : ℕ+), 
  2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_squares_representation_l2138_213880


namespace NUMINAMATH_CALUDE_three_pairs_product_l2138_213816

theorem three_pairs_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/502 := by
  sorry

end NUMINAMATH_CALUDE_three_pairs_product_l2138_213816


namespace NUMINAMATH_CALUDE_function_geq_square_for_k_geq_4_l2138_213896

def is_increasing_square (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_geq_square_for_k_geq_4
  (f : ℕ+ → ℝ)
  (h_increasing : is_increasing_square f)
  (h_f4 : f 4 = 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
sorry

end NUMINAMATH_CALUDE_function_geq_square_for_k_geq_4_l2138_213896


namespace NUMINAMATH_CALUDE_extreme_points_theorem_l2138_213833

open Real

/-- The function f(x) = x ln x - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2

/-- Predicate indicating that f has two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂))

/-- The main theorem -/
theorem extreme_points_theorem :
  (∀ a : ℝ, has_two_extreme_points a → 0 < a ∧ a < 1/2) ∧
  (∃ a : ℝ, has_two_extreme_points a ∧
    ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
      (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
      x₁ + x₂ = x₂ / x₁) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_theorem_l2138_213833


namespace NUMINAMATH_CALUDE_binomial_square_example_l2138_213829

theorem binomial_square_example : 34^2 + 2*(34*5) + 5^2 = 1521 := by sorry

end NUMINAMATH_CALUDE_binomial_square_example_l2138_213829


namespace NUMINAMATH_CALUDE_duck_count_relation_l2138_213803

theorem duck_count_relation :
  ∀ (muscovy cayuga khaki : ℕ),
    muscovy = 39 →
    muscovy = cayuga + 4 →
    muscovy + cayuga + khaki = 90 →
    muscovy = 2 * cayuga - 31 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_count_relation_l2138_213803


namespace NUMINAMATH_CALUDE_sequence_sum_l2138_213834

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = (1 : ℝ) * (1 / 3) ^ n) →
  (a 1 = 1) →
  ∀ n : ℕ, n ≥ 1 → a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_l2138_213834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2138_213891

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_4th : a 4 = 1) : 
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2138_213891


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_with_one_eighth_property_l2138_213860

theorem no_four_digit_numbers_with_one_eighth_property : 
  ¬∃ (N : ℕ), 
    (1000 ≤ N ∧ N < 10000) ∧ 
    (∃ (a x : ℕ), 
      1 ≤ a ∧ a ≤ 9 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      x = N / 8) := by
sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_with_one_eighth_property_l2138_213860


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2138_213813

def is_hcf (a b h : ℕ) : Prop := Nat.gcd a b = h

def is_lcm (a b l : ℕ) : Prop := Nat.lcm a b = l

theorem lcm_factor_proof (A B : ℕ) 
  (h1 : is_hcf A B 23)
  (h2 : A = 322)
  (h3 : ∃ x : ℕ, is_lcm A B (23 * 13 * x)) :
  ∃ x : ℕ, is_lcm A B (23 * 13 * x) ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2138_213813


namespace NUMINAMATH_CALUDE_books_bought_is_difference_melanie_books_bought_l2138_213849

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_is_difference (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) :
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Melanie's initial number of books -/
def melanie_initial_books : ℕ := 41

/-- Melanie's final number of books -/
def melanie_final_books : ℕ := 87

/-- Theorem proving the number of books Melanie bought at the yard sale -/
theorem melanie_books_bought : 
  books_bought melanie_initial_books melanie_final_books = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_books_bought_is_difference_melanie_books_bought_l2138_213849


namespace NUMINAMATH_CALUDE_probability_standard_weight_l2138_213897

theorem probability_standard_weight (total_students : ℕ) (standard_weight_students : ℕ) :
  total_students = 500 →
  standard_weight_students = 350 →
  (standard_weight_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_standard_weight_l2138_213897


namespace NUMINAMATH_CALUDE_rotated_line_equation_l2138_213804

/-- Given a line l₁ with equation x - y - 3 = 0 rotated counterclockwise by 15° around
    the point (3,0) to obtain line l₂, the equation of l₂ is √3x - y - 3√3 = 0 --/
theorem rotated_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - y - 3 = 0
  let rotation_angle : ℝ := 15 * π / 180
  let rotation_center : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := fun x y ↦
    ∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧
    x - 3 = (x₀ - 3) * Real.cos rotation_angle - (y₀ - 0) * Real.sin rotation_angle ∧
    y - 0 = (x₀ - 3) * Real.sin rotation_angle + (y₀ - 0) * Real.cos rotation_angle
  l₂ x y ↔ Real.sqrt 3 * x - y - 3 * Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l2138_213804


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l2138_213818

/-- 
Given a spherical ball that leaves a circular hole when removed from a frozen surface,
this theorem proves that if the hole has a diameter of 30 cm and a depth of 10 cm,
then the radius of the ball is 16.25 cm.
-/
theorem ball_radius_from_hole_dimensions (hole_diameter : ℝ) (hole_depth : ℝ) 
    (h_diameter : hole_diameter = 30) 
    (h_depth : hole_depth = 10) : 
    ∃ (ball_radius : ℝ), ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l2138_213818


namespace NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l2138_213884

-- Define the triangles
def triangle_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def triangle_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Define a function to calculate the area of a right triangle
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b : ℚ) / 2

-- Theorem statement
theorem area_ratio_GHI_JKL :
  let (g1, g2, _) := triangle_GHI
  let (j1, j2, _) := triangle_JKL
  (area_right_triangle g1 g2) / (area_right_triangle j1 j2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l2138_213884


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l2138_213812

theorem rectangle_shorter_side_length 
  (perimeter : ℝ) 
  (longer_side : ℝ) 
  (h1 : perimeter = 100) 
  (h2 : longer_side = 28) : 
  (perimeter - 2 * longer_side) / 2 = 22 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l2138_213812


namespace NUMINAMATH_CALUDE_opposite_grey_is_violet_l2138_213847

-- Define the colors
inductive Color
| Yellow
| Grey
| Orange
| Violet
| Blue
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : Fin 6 → Face

-- Define a view of the cube
structure View where
  top : Color
  front : Color
  right : Color

-- Define the given views
def view1 : View := { top := Color.Yellow, front := Color.Blue, right := Color.Black }
def view2 : View := { top := Color.Orange, front := Color.Yellow, right := Color.Black }
def view3 : View := { top := Color.Orange, front := Color.Violet, right := Color.Black }

-- Theorem statement
theorem opposite_grey_is_violet (c : Cube) 
  (h1 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view1.top } ∧ 
                               c.faces f2 = { color := view1.front } ∧ 
                               c.faces f3 = { color := view1.right })
  (h2 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view2.top } ∧ 
                               c.faces f2 = { color := view2.front } ∧ 
                               c.faces f3 = { color := view2.right })
  (h3 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view3.top } ∧ 
                               c.faces f2 = { color := view3.front } ∧ 
                               c.faces f3 = { color := view3.right })
  (h4 : ∃! (f : Fin 6), c.faces f = { color := Color.Grey }) :
  ∃ (f1 f2 : Fin 6), c.faces f1 = { color := Color.Grey } ∧ 
                     c.faces f2 = { color := Color.Violet } ∧ 
                     f1 ≠ f2 ∧ 
                     ∀ (f3 : Fin 6), f3 ≠ f1 ∧ f3 ≠ f2 → 
                       (c.faces f3).color ≠ Color.Grey ∧ (c.faces f3).color ≠ Color.Violet :=
by
  sorry


end NUMINAMATH_CALUDE_opposite_grey_is_violet_l2138_213847


namespace NUMINAMATH_CALUDE_factorization_problem_1_l2138_213888

theorem factorization_problem_1 (x : ℝ) : -27 + 3 * x^2 = -3 * (3 + x) * (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l2138_213888


namespace NUMINAMATH_CALUDE_bed_fraction_of_plot_l2138_213877

/-- Given a square plot of land with side length 8 units, prove that the fraction
    of the plot occupied by 13 beds (12 in an outer band and 1 central square)
    is 15/32 of the total area. -/
theorem bed_fraction_of_plot (plot_side : ℝ) (total_beds : ℕ) 
  (outer_beds : ℕ) (inner_bed_side : ℝ) :
  plot_side = 8 →
  total_beds = 13 →
  outer_beds = 12 →
  inner_bed_side = 4 →
  (outer_beds * (plot_side - inner_bed_side) + inner_bed_side ^ 2 / 2) / plot_side ^ 2 = 15 / 32 := by
  sorry

#check bed_fraction_of_plot

end NUMINAMATH_CALUDE_bed_fraction_of_plot_l2138_213877


namespace NUMINAMATH_CALUDE_min_women_proof_l2138_213842

/-- The probability of at least 4 men standing together given x women -/
def probability (x : ℕ) : ℚ :=
  (2 * Nat.choose (x + 1) 2 + (x + 1)) / (Nat.choose (x + 1) 3 + 3 * Nat.choose (x + 1) 2 + (x + 1))

/-- The minimum number of women required -/
def min_women : ℕ := 594

theorem min_women_proof :
  ∀ x : ℕ, x ≥ min_women ↔ probability x ≤ 1/100 := by
  sorry

#check min_women_proof

end NUMINAMATH_CALUDE_min_women_proof_l2138_213842


namespace NUMINAMATH_CALUDE_min_tangent_length_l2138_213881

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l2138_213881


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l2138_213838

theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (y = 1 - x ∧ 0 ≤ x ∧ x ≤ 1) ↔
  (ρ = 1 / (Real.cos θ + Real.sin θ) ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l2138_213838


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l2138_213873

theorem inequality_solution_equivalence (f : ℝ → ℝ) :
  (∃ x : ℝ, f x > 0) ↔ (∃ x₁ : ℝ, f x₁ > 0) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l2138_213873


namespace NUMINAMATH_CALUDE_subset_polynomial_equivalence_l2138_213824

theorem subset_polynomial_equivalence (n : ℕ) (h : n > 4) :
  (∀ (A B : Set (Fin n)), ∃ (f : Polynomial ℤ),
    (∀ a ∈ A, ∃ b ∈ B, f.eval a ≡ b [ZMOD n]) ∨
    (∀ b ∈ B, ∃ a ∈ A, f.eval b ≡ a [ZMOD n])) ↔
  Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_subset_polynomial_equivalence_l2138_213824


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2138_213870

theorem min_value_squared_sum (a b t : ℝ) (h : a + b = t) :
  (∀ x y : ℝ, x + y = t → (a^2 + 1)^2 + (b^2 + 1)^2 ≤ (x^2 + 1)^2 + (y^2 + 1)^2) →
  (a^2 + 1)^2 + (b^2 + 1)^2 = (t^4 + 8*t^2 + 16) / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2138_213870


namespace NUMINAMATH_CALUDE_A_intersect_B_l2138_213879

def A : Set ℝ := {x | (2*x - 6) / (x + 1) ≤ 0}
def B : Set ℝ := {-2, -1, 0, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2138_213879


namespace NUMINAMATH_CALUDE_finite_decimal_is_rational_l2138_213800

theorem finite_decimal_is_rational (x : ℝ) (h : ∃ (n : ℕ) (m : ℤ), x = m / (10 ^ n)) : 
  ∃ (p q : ℤ), x = p / q ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_is_rational_l2138_213800


namespace NUMINAMATH_CALUDE_green_light_probability_is_five_twelfths_l2138_213811

/-- Represents the duration of each light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Represents the cycle time of the traffic light in seconds -/
def cycleDuration (d : TrafficLightDuration) : ℕ :=
  d.red + d.green + d.yellow

/-- The probability of seeing a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (cycleDuration d)

/-- Theorem stating the probability of seeing a green light
    given specific durations for each light color -/
theorem green_light_probability_is_five_twelfths :
  let d : TrafficLightDuration := { red := 30, green := 25, yellow := 5 }
  greenLightProbability d = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_is_five_twelfths_l2138_213811


namespace NUMINAMATH_CALUDE_min_value_theorem_l2138_213874

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 3)⁻¹ ^ (1/3 : ℝ) + (y + 3)⁻¹ ^ (1/3 : ℝ) = 1/2) :
  x + 3*y ≥ 4*(1 + 3^(1/3 : ℝ))^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2138_213874


namespace NUMINAMATH_CALUDE_gcd_of_78_and_182_l2138_213856

theorem gcd_of_78_and_182 : Nat.gcd 78 182 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_78_and_182_l2138_213856


namespace NUMINAMATH_CALUDE_age_comparison_l2138_213814

theorem age_comparison (present_age : ℕ) (years_ago : ℕ) : 
  (present_age = 50) →
  (present_age = (125 * (present_age - years_ago)) / 100) →
  (present_age = (250 * present_age) / (250 + 50)) →
  (years_ago = 10) := by
sorry


end NUMINAMATH_CALUDE_age_comparison_l2138_213814


namespace NUMINAMATH_CALUDE_sqrt_integer_part_problem_l2138_213801

theorem sqrt_integer_part_problem :
  ∃ n : ℕ, 
    (∀ k : ℕ, k < 35 → ⌊Real.sqrt (n^2 + k)⌋ = n) ∧ 
    (∀ m : ℕ, m > n → ∃ j : ℕ, j < 35 ∧ ⌊Real.sqrt (m^2 + j)⌋ ≠ m) :=
sorry

end NUMINAMATH_CALUDE_sqrt_integer_part_problem_l2138_213801


namespace NUMINAMATH_CALUDE_sum_ages_after_ten_years_l2138_213836

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Prove that given Ann is 6 years old and Tom is twice her age, the sum of their ages 10 years later will be 38 years. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_after_ten_years_l2138_213836


namespace NUMINAMATH_CALUDE_f_of_f_of_3_l2138_213867

def f (x : ℝ) : ℝ := 3 * x^2 + x - 4

theorem f_of_f_of_3 : f (f 3) = 2050 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_3_l2138_213867


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_m_l2138_213819

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 4) (m + 2)
  is_pure_imaginary z → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_m_l2138_213819


namespace NUMINAMATH_CALUDE_emily_score_emily_score_proof_l2138_213876

/-- Calculates Emily's score in a dodgeball game -/
theorem emily_score (total_players : ℕ) (total_points : ℕ) (other_player_score : ℕ) : ℕ :=
  let other_players := total_players - 1
  let other_players_total := other_players * other_player_score
  total_points - other_players_total

/-- Proves Emily's score given the game conditions -/
theorem emily_score_proof :
  emily_score 8 39 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_emily_score_emily_score_proof_l2138_213876


namespace NUMINAMATH_CALUDE_range_of_m_l2138_213807

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2) →
  (∃ a b : ℝ, a < b ∧ (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2))) ∧
  (∀ a b : ℝ, (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2)) → a = 1 ∧ b = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2138_213807
