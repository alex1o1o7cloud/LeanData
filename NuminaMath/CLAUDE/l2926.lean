import Mathlib

namespace NUMINAMATH_CALUDE_green_blue_difference_l2926_292674

/-- Represents the number of parts for each color in the ratio --/
structure ColorRatio :=
  (blue : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (red : ℕ)

/-- Calculates the total number of parts in the ratio --/
def totalParts (ratio : ColorRatio) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red

/-- Calculates the number of disks for a given color based on the ratio and total disks --/
def disksPerColor (ratio : ColorRatio) (color : ℕ) (totalDisks : ℕ) : ℕ :=
  color * (totalDisks / totalParts ratio)

theorem green_blue_difference (totalDisks : ℕ) (ratio : ColorRatio) :
  totalDisks = 180 →
  ratio = ⟨3, 7, 8, 9⟩ →
  disksPerColor ratio ratio.green totalDisks - disksPerColor ratio ratio.blue totalDisks = 35 :=
by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l2926_292674


namespace NUMINAMATH_CALUDE_calculation_proof_l2926_292690

theorem calculation_proof : 
  (5 : ℚ) / 19 * ((3 + 4 / 5) * (5 + 1 / 3) + (4 + 2 / 3) * (19 / 5)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2926_292690


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2926_292696

/-- The coefficient of x^2 in the original function -/
def α : ℝ := 3

/-- The coefficient of x in the original function -/
def β : ℝ := -2

/-- The constant term in the original function -/
def γ : ℝ := 4

/-- The horizontal shift of the graph (to the left) -/
def h : ℝ := 2

/-- The vertical shift of the graph (upwards) -/
def k : ℝ := 5

/-- The coefficient of x^2 in the transformed function -/
def a : ℝ := α

/-- The coefficient of x in the transformed function -/
def b : ℝ := 2 * α * h - β

/-- The constant term in the transformed function -/
def c : ℝ := α * h^2 - β * h + γ + k

/-- Theorem stating that the sum of coefficients in the transformed function equals 30 -/
theorem sum_of_coefficients : a + b + c = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2926_292696


namespace NUMINAMATH_CALUDE_correct_weighted_mean_l2926_292610

theorem correct_weighted_mean (n : ℕ) (incorrect_mean : ℚ) 
  (error1 error2 error3 : ℚ) (w1 w2 w3 : ℕ) (n1 n2 n3 : ℕ) :
  n = 40 →
  incorrect_mean = 150 →
  error1 = 165 - 135 →
  error2 = 200 - 170 →
  error3 = 185 - 155 →
  w1 = 2 →
  w2 = 3 →
  w3 = 4 →
  n1 = 10 →
  n2 = 20 →
  n3 = 10 →
  n = n1 + n2 + n3 →
  let total_error := error1 + error2 + error3
  let correct_sum := n * incorrect_mean + total_error
  let total_weight := n1 * w1 + n2 * w2 + n3 * w3
  let weighted_mean := correct_sum / total_weight
  weighted_mean = 50.75 := by
sorry

end NUMINAMATH_CALUDE_correct_weighted_mean_l2926_292610


namespace NUMINAMATH_CALUDE_extraneous_root_condition_l2926_292650

/-- The equation has an extraneous root when m = -4 -/
theorem extraneous_root_condition (m : ℝ) : 
  (m = -4) → 
  (∃ (x : ℝ), x ≠ 2 ∧ 
    (m / (x - 2) - (2 * x) / (2 - x) = 1) ∧
    (m / (2 - 2) - (2 * 2) / (2 - 2) ≠ 1)) :=
by sorry


end NUMINAMATH_CALUDE_extraneous_root_condition_l2926_292650


namespace NUMINAMATH_CALUDE_limit_at_negative_one_l2926_292609

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem limit_at_negative_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |((f (-1 + Δx) - f (-1)) / Δx) - (-2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_negative_one_l2926_292609


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2926_292646

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Theorem: The intersection point of the parabola y = x^2 - 3x + 2 with the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), f 0 = y ∧ y = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2926_292646


namespace NUMINAMATH_CALUDE_time_to_install_one_window_l2926_292681

theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_for_remaining = 36)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_to_install_one_window_l2926_292681


namespace NUMINAMATH_CALUDE_derivative_extrema_l2926_292614

-- Define the function
def f (x : ℝ) := x^4 - 6*x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) := 4*x^3 - 12*x

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem derivative_extrema :
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≤ f' x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≥ f' x) ∧
  (∃ x ∈ interval, f' x = 72) ∧
  (∃ x ∈ interval, f' x = -8) :=
sorry

end NUMINAMATH_CALUDE_derivative_extrema_l2926_292614


namespace NUMINAMATH_CALUDE_course_selection_ways_l2926_292695

theorem course_selection_ways (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) : 
  type_a = 4 → type_b = 2 → total_selection = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2) + (Nat.choose type_a 2 * Nat.choose type_b 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_ways_l2926_292695


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2926_292611

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) - 2*i*(3 - 2*i) + (1 + i)*(2 + i) = 9 - 9*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2926_292611


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2926_292658

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n)
  (h2 : perp m n) 
  (h3 : perp_plane n α) 
  (h4 : ¬ subset m α) : 
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2926_292658


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2926_292647

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, 
  s > 0 → 
  s + 6 > 21 → 
  s + 21 > 6 → 
  6 + 21 > s → 
  54 > 6 + 21 + s ∧ 
  ∀ n : ℕ, n < 54 → ∃ t : ℝ, t > 0 ∧ t + 6 > 21 ∧ t + 21 > 6 ∧ 6 + 21 > t ∧ n ≤ 6 + 21 + t :=
by sorry

#check triangle_perimeter_bound

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2926_292647


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2926_292667

theorem trigonometric_identity (A B C : Real) (h : A + B + C = π) :
  Real.sin A * Real.cos B * Real.cos C + 
  Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = 
  Real.sin A * Real.sin B * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2926_292667


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2926_292618

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2926_292618


namespace NUMINAMATH_CALUDE_find_d_l2926_292623

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧ 
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0) ∧
  d = 3.2 := by
sorry

end NUMINAMATH_CALUDE_find_d_l2926_292623


namespace NUMINAMATH_CALUDE_mike_ride_distance_l2926_292684

/-- Represents the taxi fare structure -/
structure TaxiFare where
  start_fee : ℝ
  per_mile_fee : ℝ
  bridge_toll : ℝ

/-- Calculates the total fare for a given distance -/
def total_fare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_fee * distance + fare.bridge_toll

theorem mike_ride_distance (mike_fare annie_fare : TaxiFare) 
  (annie_distance : ℝ) (h1 : mike_fare.start_fee = 2.5) 
  (h2 : mike_fare.per_mile_fee = 0.25) (h3 : mike_fare.bridge_toll = 0)
  (h4 : annie_fare.start_fee = 2.5) (h5 : annie_fare.per_mile_fee = 0.25) 
  (h6 : annie_fare.bridge_toll = 5) (h7 : annie_distance = 26) :
  ∃ mike_distance : ℝ, 
    total_fare mike_fare mike_distance = total_fare annie_fare annie_distance ∧ 
    mike_distance = 46 := by
  sorry

end NUMINAMATH_CALUDE_mike_ride_distance_l2926_292684


namespace NUMINAMATH_CALUDE_autumn_pencil_count_l2926_292627

/-- Calculates the final number of pencils Autumn has -/
def final_pencil_count (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn's final pencil count is correct -/
theorem autumn_pencil_count :
  final_pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencil_count_l2926_292627


namespace NUMINAMATH_CALUDE_pencil_distribution_l2926_292672

theorem pencil_distribution (total_pens : Nat) (total_pencils : Nat) (max_students : Nat) :
  total_pens = 1001 →
  total_pencils = 910 →
  max_students = 91 →
  (∃ (students : Nat), students ≤ max_students ∧ 
    total_pens % students = 0 ∧ 
    total_pencils % students = 0) →
  total_pencils / max_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2926_292672


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l2926_292676

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < 14, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) ∧ 
  (Nat.gcd 14 70 > 1 ∧ Nat.gcd 14 84 > 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l2926_292676


namespace NUMINAMATH_CALUDE_edward_lives_left_l2926_292644

theorem edward_lives_left (initial_lives : ℕ) (lives_lost : ℕ) : 
  initial_lives = 15 → lives_lost = 8 → initial_lives - lives_lost = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_lives_left_l2926_292644


namespace NUMINAMATH_CALUDE_max_min_value_l2926_292694

theorem max_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  ∃ (max_s : ℝ), max_s = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → min x' (min (y' + 1/x') (1/y')) ≤ max_s) ∧
    (s = max_s ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_l2926_292694


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2926_292685

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2926_292685


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2926_292648

-- Define the condition for a line to be tangent to the circle
def is_tangent (k : ℝ) : Prop :=
  1 + k^2 = 4

-- Define the main theorem
theorem sufficient_not_necessary :
  (∀ k, k = Real.sqrt 3 → is_tangent k) ∧
  (∃ k, k ≠ Real.sqrt 3 ∧ is_tangent k) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2926_292648


namespace NUMINAMATH_CALUDE_time_walking_away_l2926_292651

-- Define the walking speed in miles per hour
def walking_speed : ℝ := 2

-- Define the total distance walked in miles
def total_distance : ℝ := 12

-- Define the theorem
theorem time_walking_away : 
  (total_distance / 2) / walking_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_walking_away_l2926_292651


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2926_292607

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 4) : 
  x^4 + y^4 = 8432 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2926_292607


namespace NUMINAMATH_CALUDE_probability_king_of_diamonds_l2926_292688

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The game setup with two standard decks -/
def game_setup (d : Deck) : Prop :=
  d.cards = 52 ∧ d.ranks = 13 ∧ d.suits = 4

/-- The total number of cards in the combined deck -/
def total_cards (d : Deck) : Nat :=
  2 * d.cards

/-- The number of Kings of Diamonds in the combined deck -/
def kings_of_diamonds : Nat := 2

/-- The probability of drawing a King of Diamonds from the top of the combined deck -/
theorem probability_king_of_diamonds (d : Deck) :
  game_setup d →
  (kings_of_diamonds : ℚ) / (total_cards d) = 1 / 52 :=
by sorry

end NUMINAMATH_CALUDE_probability_king_of_diamonds_l2926_292688


namespace NUMINAMATH_CALUDE_officer_jawan_groups_count_l2926_292671

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 officers from 7 and 4 jawans from 12 -/
def officer_jawan_groups : ℕ :=
  binomial 7 2 * binomial 12 4

theorem officer_jawan_groups_count :
  officer_jawan_groups = 20790 := by sorry

end NUMINAMATH_CALUDE_officer_jawan_groups_count_l2926_292671


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2926_292668

def club_growth (b : ℕ → ℕ) : Prop :=
  b 0 = 20 ∧ ∀ k, b (k + 1) = 4 * b k - 12

theorem club_size_after_four_years (b : ℕ → ℕ) (h : club_growth b) : b 4 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2926_292668


namespace NUMINAMATH_CALUDE_sine_cosine_ratio_equals_tangent_l2926_292680

theorem sine_cosine_ratio_equals_tangent :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) / 
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) = 
  Real.tan (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_ratio_equals_tangent_l2926_292680


namespace NUMINAMATH_CALUDE_bethany_current_age_l2926_292665

/-- Bethany's current age -/
def bethany_age : ℕ := sorry

/-- Bethany's sister's current age -/
def sister_age : ℕ := sorry

/-- Bethany's brother's current age -/
def brother_age : ℕ := sorry

/-- Theorem stating Bethany's current age given the conditions -/
theorem bethany_current_age :
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧
  (bethany_age - 3 = brother_age - 3 + 4) ∧
  (sister_age + 5 = 16) ∧
  (brother_age + 5 = 21) →
  bethany_age = 19 := by sorry

end NUMINAMATH_CALUDE_bethany_current_age_l2926_292665


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l2926_292659

theorem roots_sum_reciprocal (α β : ℝ) : 
  α^2 - 10*α + 20 = 0 → β^2 - 10*β + 20 = 0 → 1/α + 1/β = 1/2 := by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l2926_292659


namespace NUMINAMATH_CALUDE_stating_trail_mix_theorem_l2926_292603

/-- Represents the number of bags of nuts -/
def nuts : ℕ := 16

/-- Represents the number of portions that can be made -/
def portions : ℕ := 2

/-- Represents the number of bags of dried fruit -/
def dried_fruit : ℕ := 2

/-- 
Theorem stating that given 16 bags of nuts and the constraint that the maximum 
number of equal portions is 2 with no bags left over, the number of bags of 
dried fruit must be 2.
-/
theorem trail_mix_theorem : 
  (nuts + dried_fruit) % portions = 0 ∧ 
  ∀ n : ℕ, n > dried_fruit → (nuts + n) % portions ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_stating_trail_mix_theorem_l2926_292603


namespace NUMINAMATH_CALUDE_remainder_theorem_l2926_292661

theorem remainder_theorem : ∃ q : ℤ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2926_292661


namespace NUMINAMATH_CALUDE_magazine_cost_l2926_292654

theorem magazine_cost (m : ℝ) 
  (h1 : 8 * m < 12) 
  (h2 : 11 * m > 16.5) : 
  m = 1.5 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l2926_292654


namespace NUMINAMATH_CALUDE_abc_remainder_mod_seven_l2926_292625

theorem abc_remainder_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (a + 2*b + 3*c) % 7 = 1 →
  (2*a + 3*b + c) % 7 = 2 →
  (3*a + b + 2*c) % 7 = 1 →
  (a*b*c) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_seven_l2926_292625


namespace NUMINAMATH_CALUDE_remainder_3_pow_123_plus_4_mod_8_l2926_292602

theorem remainder_3_pow_123_plus_4_mod_8 : 3^123 + 4 ≡ 7 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_123_plus_4_mod_8_l2926_292602


namespace NUMINAMATH_CALUDE_lines_relationship_l2926_292637

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for non-coplanar lines -/
def nonCoplanar (l₁ l₂ : Line3D) : Prop := sorry

/-- Predicate for intersecting lines -/
def intersects (l₁ l₂ : Line3D) : Prop := sorry

/-- Theorem: Given non-coplanar lines l₁ and l₂, and lines m₁ and m₂ that both intersect with l₁ and l₂,
    the positional relationship between m₁ and m₂ is either intersecting or non-coplanar -/
theorem lines_relationship (l₁ l₂ m₁ m₂ : Line3D)
  (h₁ : nonCoplanar l₁ l₂)
  (h₂ : intersects m₁ l₁)
  (h₃ : intersects m₁ l₂)
  (h₄ : intersects m₂ l₁)
  (h₅ : intersects m₂ l₂) :
  intersects m₁ m₂ ∨ nonCoplanar m₁ m₂ := by
  sorry

end NUMINAMATH_CALUDE_lines_relationship_l2926_292637


namespace NUMINAMATH_CALUDE_fraction_simplification_l2926_292679

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : x + 1/y ≠ 0) :
  (x + 1/y) / (y + 1/x) = x/y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2926_292679


namespace NUMINAMATH_CALUDE_vector_projection_l2926_292616

/-- Given two 2D vectors a and b, prove that the projection of a onto b is √13/13 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (-2, 1) → b = (-2, -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2926_292616


namespace NUMINAMATH_CALUDE_marias_test_score_l2926_292666

theorem marias_test_score (scores : Fin 4 → ℕ) : 
  scores 0 = 80 →
  scores 2 = 90 →
  scores 3 = 100 →
  (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 85 →
  scores 1 = 70 := by
sorry

end NUMINAMATH_CALUDE_marias_test_score_l2926_292666


namespace NUMINAMATH_CALUDE_pilot_fish_speed_is_30_l2926_292673

/-- Calculates the speed of a pilot fish given initial conditions -/
def pilotFishSpeed (keanuSpeed : ℝ) (sharkSpeedMultiplier : ℝ) (pilotFishIncreaseFactor : ℝ) : ℝ :=
  let sharkSpeedIncrease := keanuSpeed * (sharkSpeedMultiplier - 1)
  keanuSpeed + pilotFishIncreaseFactor * sharkSpeedIncrease

/-- Theorem stating that under given conditions, the pilot fish's speed is 30 mph -/
theorem pilot_fish_speed_is_30 :
  pilotFishSpeed 20 2 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pilot_fish_speed_is_30_l2926_292673


namespace NUMINAMATH_CALUDE_expression_simplification_l2926_292693

theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a - b) = (a^3 - 3*a*b + b^3) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2926_292693


namespace NUMINAMATH_CALUDE_f_properties_l2926_292612

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x - 3)

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 81) ∧
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2926_292612


namespace NUMINAMATH_CALUDE_units_digit_problem_l2926_292692

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ 
  (((2008^2 + 2^2008)^2 + 2^(2008^2 + 2^2008)) % 10 = n % 10) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2926_292692


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2926_292626

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, x^2 - x - m + 1 > 0) ↔ m < 3/4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2926_292626


namespace NUMINAMATH_CALUDE_rational_additive_function_is_linear_l2926_292632

theorem rational_additive_function_is_linear 
  (f : ℚ → ℚ) 
  (h : ∀ (x y : ℚ), f (x + y) = f x + f y) : 
  ∃ (c : ℚ), ∀ (x : ℚ), f x = c * x := by
sorry

end NUMINAMATH_CALUDE_rational_additive_function_is_linear_l2926_292632


namespace NUMINAMATH_CALUDE_cards_per_player_l2926_292691

/-- Proves that evenly distributing 54 cards among 3 players results in 18 cards per player -/
theorem cards_per_player (initial_cards : ℕ) (added_cards : ℕ) (num_players : ℕ) :
  initial_cards = 52 →
  added_cards = 2 →
  num_players = 3 →
  (initial_cards + added_cards) / num_players = 18 := by
sorry

end NUMINAMATH_CALUDE_cards_per_player_l2926_292691


namespace NUMINAMATH_CALUDE_farm_field_correct_l2926_292630

/-- Represents the farm field ploughing problem -/
structure FarmField where
  total_area : ℕ  -- Total area of the farm field in hectares
  planned_days : ℕ  -- Initially planned number of days
  daily_plan : ℕ  -- Hectares planned to be ploughed per day
  actual_daily : ℕ  -- Hectares actually ploughed per day
  extra_days : ℕ  -- Additional days worked
  remaining : ℕ  -- Hectares remaining to be ploughed

/-- The farm field problem solution -/
def farm_field_solution : FarmField :=
  { total_area := 720
  , planned_days := 6
  , daily_plan := 120
  , actual_daily := 85
  , extra_days := 2
  , remaining := 40 }

/-- Theorem stating the correctness of the farm field problem solution -/
theorem farm_field_correct (f : FarmField) : 
  f.daily_plan * f.planned_days = f.total_area ∧
  f.actual_daily * (f.planned_days + f.extra_days) + f.remaining = f.total_area ∧
  f = farm_field_solution := by
  sorry

#check farm_field_correct

end NUMINAMATH_CALUDE_farm_field_correct_l2926_292630


namespace NUMINAMATH_CALUDE_binary_11011_to_decimal_l2926_292652

def binary_to_decimal (b₄ b₃ b₂ b₁ b₀ : Nat) : Nat :=
  b₄ * 2^4 + b₃ * 2^3 + b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_11011_to_decimal :
  binary_to_decimal 1 1 0 1 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_to_decimal_l2926_292652


namespace NUMINAMATH_CALUDE_sum_digits_of_numeric_hex_count_l2926_292634

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number --/
def Hexadecimal := List HexDigit

/-- Converts a natural number to hexadecimal --/
def toHex (n : ℕ) : Hexadecimal :=
  sorry

/-- Checks if a hexadecimal number uses only numeric digits --/
def usesOnlyNumericDigits (h : Hexadecimal) : Bool :=
  sorry

/-- Counts numbers representable in hexadecimal using only numeric digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem sum_digits_of_numeric_hex_count :
  sumDigits (countNumericHex 2000) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_of_numeric_hex_count_l2926_292634


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2926_292698

theorem tic_tac_toe_tie_probability (amy_win : ℚ) (lily_win : ℚ) (john_win : ℚ)
  (h_amy : amy_win = 4/9)
  (h_lily : lily_win = 1/3)
  (h_john : john_win = 1/6) :
  1 - (amy_win + lily_win + john_win) = 1/18 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2926_292698


namespace NUMINAMATH_CALUDE_triangle_area_l2926_292664

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (2 * a * b * Real.sin C = Real.sqrt 3 * (b^2 + c^2 - a^2)) →
  (a = Real.sqrt 13) →
  (c = 3) →
  (1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2926_292664


namespace NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l2926_292678

theorem shifted_sine_equals_cosine (φ : Real) (h : 0 < φ ∧ φ < π) :
  (∀ x, 2 * Real.sin (2 * x - π / 3 + φ) = 2 * Real.cos (2 * x)) ↔ φ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l2926_292678


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2926_292621

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + m = 0 → y = x) ↔ 
  m = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2926_292621


namespace NUMINAMATH_CALUDE_bus_breakdown_time_correct_l2926_292677

/-- Represents the scenario of a school trip with a bus breakdown -/
structure BusBreakdown where
  S : ℝ  -- Distance between school and county town in km
  x : ℝ  -- Walking speed in km/minute
  t : ℝ  -- Walking time of teachers and students in minutes
  a : ℝ  -- Bus breakdown time in minutes

/-- The bus speed is 5 times the walking speed -/
def bus_speed (bd : BusBreakdown) : ℝ := 5 * bd.x

/-- The walking time satisfies the equation derived from the problem conditions -/
def walking_time_equation (bd : BusBreakdown) : Prop :=
  bd.t = bd.S / (5 * bd.x) + 20 - (bd.S - bd.x * bd.t) / (5 * bd.x)

/-- The bus breakdown time satisfies the equation derived from the problem conditions -/
def breakdown_time_equation (bd : BusBreakdown) : Prop :=
  bd.a + (2 * (bd.S - bd.x * bd.t)) / (5 * bd.x) = (2 * bd.S) / (5 * bd.x) + 30

/-- Theorem stating that given the conditions, the bus breakdown time equation holds -/
theorem bus_breakdown_time_correct (bd : BusBreakdown) 
  (h_walking_time : walking_time_equation bd) :
  breakdown_time_equation bd :=
sorry

end NUMINAMATH_CALUDE_bus_breakdown_time_correct_l2926_292677


namespace NUMINAMATH_CALUDE_incorrect_representation_l2926_292670

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  nonRepeating : ℕ → ℕ  -- P: mapping from position to digit
  repeating : ℕ → ℕ     -- Q: mapping from position to digit
  r : ℕ                 -- length of non-repeating part
  s : ℕ                 -- length of repeating part

/-- The decimal representation of a RepeatingDecimal -/
def decimalRepresentation (d : RepeatingDecimal) : ℚ :=
  sorry

/-- Theorem stating that the given representation is incorrect -/
theorem incorrect_representation (d : RepeatingDecimal) :
  ∃ (P Q : ℕ), 
    (10^d.r * 10^(2*d.s) * decimalRepresentation d ≠ 
     (P * 100 + Q * 10 + Q : ℚ) + decimalRepresentation d) :=
  sorry

end NUMINAMATH_CALUDE_incorrect_representation_l2926_292670


namespace NUMINAMATH_CALUDE_remainder_product_l2926_292635

theorem remainder_product (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (rem_a : a % 8 = 3) (rem_b : b % 6 = 5) : (a * b) % 48 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l2926_292635


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2926_292600

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2926_292600


namespace NUMINAMATH_CALUDE_sum_of_squares_of_G_digits_l2926_292620

/-- Represents a fraction m/n -/
structure Fraction where
  m : ℕ+
  n : ℕ+
  m_lt_n : m < n
  lowest_terms : Nat.gcd m n = 1
  no_square_divisor : ∀ k > 1, ¬(k * k ∣ n)
  repeating_length_6 : ∃ k : ℕ+, m * 10^6 = k * n + m

/-- Count of valid fractions -/
def F : ℕ := 1109700

/-- Number of digits in F -/
def p : ℕ := 7

/-- G is defined as F + p -/
def G : ℕ := F + p

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem to prove -/
theorem sum_of_squares_of_G_digits :
  sum_of_squares_of_digits G = 181 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_G_digits_l2926_292620


namespace NUMINAMATH_CALUDE_inverse_proportion_l2926_292662

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 10 * 3 = k) :
  -15 * -2 = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2926_292662


namespace NUMINAMATH_CALUDE_curve_tangent_sum_l2926_292636

/-- The curve C defined by y = x^3 - x^2 - ax + b -/
def C (x y a b : ℝ) : Prop := y = x^3 - x^2 - a*x + b

/-- The derivative of C with respect to x -/
def C_derivative (x a : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem curve_tangent_sum (a b : ℝ) : 
  C 0 1 a b ∧ C_derivative 0 a = 2 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_sum_l2926_292636


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l2926_292642

theorem unique_solution_cubic_equation :
  ∀ x : ℝ, (1 + x^2) * (1 + x^4) = 4 * x^3 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l2926_292642


namespace NUMINAMATH_CALUDE_rachel_essay_time_l2926_292689

/-- Rachel's essay writing process -/
def essay_time (pages_per_30min : ℕ) (research_time : ℕ) (total_pages : ℕ) (editing_time : ℕ) : ℕ :=
  let writing_time := 30 * total_pages / pages_per_30min
  (research_time + writing_time + editing_time) / 60

/-- Theorem: Rachel spends 5 hours completing her essay -/
theorem rachel_essay_time :
  essay_time 1 45 6 75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_essay_time_l2926_292689


namespace NUMINAMATH_CALUDE_correct_calculation_l2926_292682

theorem correct_calculation : (-9)^2 / (-3)^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2926_292682


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2926_292641

/-- Given two people p and q, where 8 years ago p was half of q's age,
    and the total of their present ages is 28,
    prove that the ratio of their present ages is 3:4 -/
theorem age_ratio_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) → 
  (p + q = 28) → 
  (p : ℚ) / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2926_292641


namespace NUMINAMATH_CALUDE_time_capsule_depth_relation_l2926_292624

/-- Represents the relationship between the depths of time capsules buried by Southton and Northton -/
theorem time_capsule_depth_relation (x y z : ℝ) : 
  (y = 4 * x + z) ↔ (y - 4 * x = z) :=
by sorry

end NUMINAMATH_CALUDE_time_capsule_depth_relation_l2926_292624


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_seven_l2926_292657

/-- A quadratic function f(x) = x^2 + (a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def monotonically_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 4 → f a x ≥ f a y

/-- Theorem: If f is monotonically decreasing on (-∞, 4], then a ≤ -7 -/
theorem monotonic_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  monotonically_decreasing a → a ≤ -7 := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_seven_l2926_292657


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2926_292633

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 7*x + 10) + (x^2 + 7*x + 10) =
  (x^2 + 7*x + 20) * (x^2 + 7*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2926_292633


namespace NUMINAMATH_CALUDE_andy_coat_production_l2926_292608

/-- Given the conditions about minks and coat production, prove that Andy can make 7 coats. -/
theorem andy_coat_production (
  minks_per_coat : ℕ := 15
  ) (
  initial_minks : ℕ := 30
  ) (
  babies_per_mink : ℕ := 6
  ) (
  freed_fraction : ℚ := 1/2
  ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_andy_coat_production_l2926_292608


namespace NUMINAMATH_CALUDE_total_amount_proof_l2926_292687

/-- Given that r has two-thirds of the total amount with p and q, and r has 1600,
    prove that the total amount T with p, q, and r is 4000. -/
theorem total_amount_proof (T : ℝ) (r : ℝ) 
  (h1 : r = 2/3 * T)
  (h2 : r = 1600) : 
  T = 4000 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l2926_292687


namespace NUMINAMATH_CALUDE_absolute_value_trig_expression_l2926_292645

theorem absolute_value_trig_expression : 
  |(-3 : ℝ)| + Real.sqrt 3 * Real.sin (60 * π / 180) - (1 / 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_trig_expression_l2926_292645


namespace NUMINAMATH_CALUDE_river_depth_problem_l2926_292697

/-- River depth problem -/
theorem river_depth_problem (may_depth june_depth july_depth : ℕ) : 
  may_depth = 5 →
  june_depth = may_depth + 10 →
  july_depth = june_depth * 3 →
  july_depth = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_problem_l2926_292697


namespace NUMINAMATH_CALUDE_expected_red_pairs_in_standard_deck_l2926_292640

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def redAdjacentProbability : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard deck dealt in a circle -/
def expectedRedPairs : ℚ := redCardCount * redAdjacentProbability

theorem expected_red_pairs_in_standard_deck :
  expectedRedPairs = 650 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_in_standard_deck_l2926_292640


namespace NUMINAMATH_CALUDE_unique_solution_cube_sum_l2926_292605

theorem unique_solution_cube_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_sum_l2926_292605


namespace NUMINAMATH_CALUDE_two_A_minus_three_B_two_A_minus_three_B_equals_seven_l2926_292639

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B (x y : ℝ) : ℝ := y^2 - x * y + 2 * x^2

-- Theorem 1: 2A - 3B = y² - xy
theorem two_A_minus_three_B (x y : ℝ) : 2 * A x y - 3 * B x y = y^2 - x * y := by
  sorry

-- Theorem 2: 2A - 3B = 7 under the given condition
theorem two_A_minus_three_B_equals_seven (x y : ℝ) 
  (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_A_minus_three_B_two_A_minus_three_B_equals_seven_l2926_292639


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2926_292606

theorem power_of_two_equation (x : ℕ) : 16^3 + 16^3 + 16^3 = 2^x ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2926_292606


namespace NUMINAMATH_CALUDE_root_in_interval_l2926_292601

theorem root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2926_292601


namespace NUMINAMATH_CALUDE_evaluate_expression_l2926_292699

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) :
  x^3 * y^4 * z = 1/128 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2926_292699


namespace NUMINAMATH_CALUDE_distance_AB_is_40_l2926_292683

/-- The distance between two points A and B -/
def distance_AB : ℝ := 40

/-- The remaining distance for the second cyclist when the first cyclist has traveled half the total distance -/
def remaining_distance_second : ℝ := 24

/-- The remaining distance for the first cyclist when the second cyclist has traveled half the total distance -/
def remaining_distance_first : ℝ := 15

/-- The theorem stating that the distance between points A and B is 40 km -/
theorem distance_AB_is_40 :
  (distance_AB / 2 + remaining_distance_second = distance_AB) ∧
  (distance_AB / 2 + remaining_distance_first = distance_AB) →
  distance_AB = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_40_l2926_292683


namespace NUMINAMATH_CALUDE_problem_solution_l2926_292622

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem problem_solution (a : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence (fun n => a (n + 1) - a n) →
  (fun n => a (n + 1) - a n) 1 = 2 →
  (∀ n : ℕ, (fun n => a (n + 1) - a n) (n + 1) - (fun n => a (n + 1) - a n) n = 2) →
  a 1 = 1 →
  43 < a m →
  a m < 73 →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2926_292622


namespace NUMINAMATH_CALUDE_spelling_bee_points_l2926_292649

theorem spelling_bee_points (max_points : ℕ) : max_points = 5 := by
  -- Define Dulce's points
  let dulce_points : ℕ := 3

  -- Define Val's points in terms of Max and Dulce's points
  let val_points : ℕ := 2 * (max_points + dulce_points)

  -- Define the total points of Max's team
  let team_points : ℕ := max_points + dulce_points + val_points

  -- Define the opponents' team points
  let opponents_points : ℕ := 40

  -- Express that Max's team is behind by 16 points
  have team_difference : team_points = opponents_points - 16 := by sorry

  -- Prove that max_points = 5
  sorry

end NUMINAMATH_CALUDE_spelling_bee_points_l2926_292649


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_l2926_292631

theorem bicycle_helmet_cost (helmet_cost bicycle_cost total_cost : ℕ) : 
  helmet_cost = 40 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  total_cost = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_l2926_292631


namespace NUMINAMATH_CALUDE_fifth_root_division_l2926_292686

theorem fifth_root_division (x : ℝ) (h : x > 0) :
  (x ^ (1 / 3)) / (x ^ (1 / 5)) = x ^ (2 / 15) :=
sorry

end NUMINAMATH_CALUDE_fifth_root_division_l2926_292686


namespace NUMINAMATH_CALUDE_power_of_power_three_l2926_292653

theorem power_of_power_three : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2926_292653


namespace NUMINAMATH_CALUDE_power_of_product_rule_l2926_292669

theorem power_of_product_rule (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_rule_l2926_292669


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2926_292619

theorem quadratic_equation_roots (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + k = 0 ∧ x₂^2 - x₂ + k = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2926_292619


namespace NUMINAMATH_CALUDE_circle_slope_range_l2926_292663

/-- The range of y/x for points on the circle x^2 + y^2 - 4x - 6y + 12 = 0 -/
theorem circle_slope_range :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 6*p.2 + 12 = 0}
  ∀ (x y : ℝ), (x, y) ∈ circle → x ≠ 0 →
    (6 - 2*Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2*Real.sqrt 3) / 3 :=
by sorry


end NUMINAMATH_CALUDE_circle_slope_range_l2926_292663


namespace NUMINAMATH_CALUDE_matching_pair_probability_l2926_292660

def black_socks : ℕ := 12
def blue_socks : ℕ := 10

def total_socks : ℕ := black_socks + blue_socks

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def total_ways : ℕ := choose total_socks 2
def black_matching_ways : ℕ := choose black_socks 2
def blue_matching_ways : ℕ := choose blue_socks 2
def matching_ways : ℕ := black_matching_ways + blue_matching_ways

theorem matching_pair_probability :
  (matching_ways : ℚ) / total_ways = 111 / 231 :=
sorry

end NUMINAMATH_CALUDE_matching_pair_probability_l2926_292660


namespace NUMINAMATH_CALUDE_remaining_legos_l2926_292675

theorem remaining_legos (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 2080 → lost = 17 → remaining = initial - lost → remaining = 2063 := by
sorry

end NUMINAMATH_CALUDE_remaining_legos_l2926_292675


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2926_292656

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 4.5 →
  time = 4 →
  interest = 144 →
  principal * rate * time / 100 = interest →
  principal = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2926_292656


namespace NUMINAMATH_CALUDE_coffee_maker_discount_l2926_292638

/-- Calculates the discount amount given the original price and discounted price. -/
def discount_amount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Proves that the discount amount is 20 dollars for a coffee maker with an original price
    of 90 dollars and a discounted price of 70 dollars. -/
theorem coffee_maker_discount : discount_amount 90 70 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_discount_l2926_292638


namespace NUMINAMATH_CALUDE_count_good_numbers_formula_l2926_292613

/-- A number is considered "good" if it contains an even number (including zero) of the digit 8 -/
def is_good (x : ℕ) : Prop := sorry

/-- The count of "good numbers" with length not exceeding n -/
def count_good_numbers (n : ℕ) : ℕ := sorry

/-- The main theorem: The count of "good numbers" with length not exceeding n 
    is equal to (8^n + 10^n) / 2 - 1 -/
theorem count_good_numbers_formula (n : ℕ) (h : n > 0) : 
  count_good_numbers n = (8^n + 10^n) / 2 - 1 := by sorry

end NUMINAMATH_CALUDE_count_good_numbers_formula_l2926_292613


namespace NUMINAMATH_CALUDE_shifted_function_point_l2926_292615

/-- A function whose graph passes through (1, -1) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 1 = -1

/-- The horizontally shifted function -/
def shift_function (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x - 3)

theorem shifted_function_point (f : ℝ → ℝ) :
  passes_through_point f → passes_through_point (shift_function f) :=
by
  sorry

#check shifted_function_point

end NUMINAMATH_CALUDE_shifted_function_point_l2926_292615


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_zero_l2926_292655

theorem sum_of_odd_powers_zero (a b : ℝ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  a^(2*n+1) + b^(2*n+1) = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_zero_l2926_292655


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2926_292628

theorem sqrt_equation_solutions (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3) ↔
  (x = 32.25 ∨ x = 8.25) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2926_292628


namespace NUMINAMATH_CALUDE_sqrt_72_simplification_l2926_292629

theorem sqrt_72_simplification : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_simplification_l2926_292629


namespace NUMINAMATH_CALUDE_certain_number_problem_l2926_292643

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 300 → N = 576 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2926_292643


namespace NUMINAMATH_CALUDE_triangle_problem_l2926_292604

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) :
  (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0 →
  (1 / 2) * t.a * t.c * Real.sin t.B = 15 * Real.sqrt 3 →
  t.a + t.b + t.c = 30 →
  Real.sin (2 * t.B) / (Real.sin t.A + Real.sin t.C) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2926_292604


namespace NUMINAMATH_CALUDE_power_equality_l2926_292617

theorem power_equality : 32^3 * 8^4 = 2^27 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2926_292617
