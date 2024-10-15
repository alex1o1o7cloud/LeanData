import Mathlib

namespace NUMINAMATH_CALUDE_balance_theorem_l3657_365717

/-- Represents the weight of a ball in terms of blue balls -/
@[ext] structure BallWeight where
  blue : ℚ

/-- The weight of a red ball in terms of blue balls -/
def red_weight : BallWeight := ⟨2⟩

/-- The weight of a yellow ball in terms of blue balls -/
def yellow_weight : BallWeight := ⟨3⟩

/-- The weight of a white ball in terms of blue balls -/
def white_weight : BallWeight := ⟨5/3⟩

/-- The weight of a blue ball in terms of blue balls -/
def blue_weight : BallWeight := ⟨1⟩

theorem balance_theorem :
  2 * red_weight.blue + 4 * yellow_weight.blue + 3 * white_weight.blue = 21 * blue_weight.blue :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l3657_365717


namespace NUMINAMATH_CALUDE_crates_in_third_trip_is_two_l3657_365777

/-- Represents the problem of distributing crates across multiple trips. -/
structure CrateDistribution where
  total_crates : ℕ
  min_crate_weight : ℕ
  max_trip_weight : ℕ

/-- Calculates the number of crates in the third trip. -/
def crates_in_third_trip (cd : CrateDistribution) : ℕ :=
  cd.total_crates - 2 * (cd.max_trip_weight / cd.min_crate_weight)

/-- Theorem stating that for the given conditions, the number of crates in the third trip is 2. -/
theorem crates_in_third_trip_is_two :
  let cd : CrateDistribution := {
    total_crates := 12,
    min_crate_weight := 120,
    max_trip_weight := 600
  }
  crates_in_third_trip cd = 2 := by
  sorry

end NUMINAMATH_CALUDE_crates_in_third_trip_is_two_l3657_365777


namespace NUMINAMATH_CALUDE_gcd_228_1995_l3657_365702

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l3657_365702


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l3657_365796

def M : Nat := 2^5 * 3^4 * 5^3 * 11^2

theorem number_of_factors_of_M : 
  (Finset.filter (·∣M) (Finset.range (M + 1))).card = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l3657_365796


namespace NUMINAMATH_CALUDE_lines_intersect_l3657_365736

/-- Represents a line in the form Ax + By + C = 0 --/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if two lines are intersecting --/
def are_intersecting (l1 l2 : Line) : Prop :=
  l1.A * l2.B ≠ l2.A * l1.B

theorem lines_intersect : 
  let line1 : Line := { A := 3, B := -2, C := 5 }
  let line2 : Line := { A := 1, B := 3, C := 10 }
  are_intersecting line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_l3657_365736


namespace NUMINAMATH_CALUDE_correct_calculation_l3657_365722

theorem correct_calculation : 
  (67 * 17 ≠ 1649) ∧ 
  (150 * 60 ≠ 900) ∧ 
  (250 * 70 = 17500) ∧ 
  (98 * 36 ≠ 3822) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l3657_365722


namespace NUMINAMATH_CALUDE_castle_provisions_duration_l3657_365760

/-- 
Proves that given the conditions of the castle's food provisions,
the initial food supply was meant to last 120 days.
-/
theorem castle_provisions_duration 
  (initial_people : ℕ) 
  (people_left : ℕ) 
  (days_before_leaving : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_people = 300)
  (h2 : people_left = 100)
  (h3 : days_before_leaving = 30)
  (h4 : days_after_leaving = 90)
  : ℕ := by
  sorry

#check castle_provisions_duration

end NUMINAMATH_CALUDE_castle_provisions_duration_l3657_365760


namespace NUMINAMATH_CALUDE_whole_number_between_l3657_365728

theorem whole_number_between : 
  ∃ (M : ℕ), (8 : ℚ) < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9 → M = 33 :=
by sorry

end NUMINAMATH_CALUDE_whole_number_between_l3657_365728


namespace NUMINAMATH_CALUDE_track_length_l3657_365735

/-- The length of a track AB given specific meeting points of two athletes --/
theorem track_length (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  let x := (v₁ + v₂) * 300 / v₂
  (300 / v₁ = (x - 300) / v₂) ∧ ((x + 100) / v₁ = (x - 100) / v₂) → x = 500 := by
  sorry

#check track_length

end NUMINAMATH_CALUDE_track_length_l3657_365735


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l3657_365768

/-- The maximum distance from any point on the unit circle to the line x - y + 3 = 0 -/
theorem max_distance_circle_to_line : 
  ∃ (d : ℝ), d = (3 * Real.sqrt 2) / 2 + 1 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  |x - y + 3| / Real.sqrt 2 ≤ d ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ |x₀ - y₀ + 3| / Real.sqrt 2 = d :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l3657_365768


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3657_365790

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, (-3 ≤ x ∧ x ≤ 1) → (x ≤ 2 ∨ x ≥ 3)) ∧
  (∃ x : ℝ, (x ≤ 2 ∨ x ≥ 3) ∧ ¬(-3 ≤ x ∧ x ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3657_365790


namespace NUMINAMATH_CALUDE_goods_train_speed_calculation_l3657_365799

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 40

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 12

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 350

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 65

/-- Theorem stating that the given conditions imply the correct speed of the goods train -/
theorem goods_train_speed_calculation :
  man_train_speed = 40 ∧
  passing_time = 12 ∧
  goods_train_length = 350 →
  goods_train_speed = 65 := by
  sorry

#check goods_train_speed_calculation

end NUMINAMATH_CALUDE_goods_train_speed_calculation_l3657_365799


namespace NUMINAMATH_CALUDE_range_of_t_l3657_365746

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x ≤ t ∧ x^2 - 4*x + t ≤ 0) → 
  0 ≤ t ∧ t ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l3657_365746


namespace NUMINAMATH_CALUDE_doug_lost_marbles_l3657_365714

theorem doug_lost_marbles (d : ℕ) (l : ℕ) : 
  (d + 22 = d - l + 30) → l = 8 := by
  sorry

end NUMINAMATH_CALUDE_doug_lost_marbles_l3657_365714


namespace NUMINAMATH_CALUDE_journey_time_ratio_l3657_365718

/-- Proves that the ratio of the time taken for the journey back to the time taken for the journey to San Francisco is 3:2, given the average speeds -/
theorem journey_time_ratio (distance : ℝ) (speed_to_sf : ℝ) (avg_speed : ℝ)
  (h1 : speed_to_sf = 45)
  (h2 : avg_speed = 30)
  (h3 : distance > 0) :
  (distance / avg_speed - distance / speed_to_sf) / (distance / speed_to_sf) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l3657_365718


namespace NUMINAMATH_CALUDE_max_hollow_cube_volume_l3657_365788

/-- The number of available unit cubes --/
def available_cubes : ℕ := 1000

/-- Function to calculate the number of cubes used for a given side length --/
def cubes_used (x : ℕ) : ℕ :=
  2 * x^2 + 2 * x * (x - 2) + 2 * (x - 2)^2

/-- The maximum side length that can be achieved --/
def max_side_length : ℕ := 13

/-- Theorem stating the maximum volume that can be achieved --/
theorem max_hollow_cube_volume :
  (∀ x : ℕ, cubes_used x ≤ available_cubes → x ≤ max_side_length) ∧
  cubes_used max_side_length ≤ available_cubes ∧
  max_side_length^3 = 2197 :=
sorry

end NUMINAMATH_CALUDE_max_hollow_cube_volume_l3657_365788


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_plane_perp_l3657_365713

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_plane_perp 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_plane_perp_l3657_365713


namespace NUMINAMATH_CALUDE_product_990_sum_93_l3657_365745

theorem product_990_sum_93 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 990) ∧ 
  (x * y * z = 990) ∧ 
  (a + b + x + y + z = 93) := by
sorry

end NUMINAMATH_CALUDE_product_990_sum_93_l3657_365745


namespace NUMINAMATH_CALUDE_range_of_a_l3657_365740

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : even_function f)
  (h_incr : increasing_on_neg f)
  (h_cond : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3657_365740


namespace NUMINAMATH_CALUDE_line_inclination_45_implies_a_equals_1_l3657_365703

/-- If the line ax + (2a - 3)y = 0 has an angle of inclination of 45°, then a = 1 -/
theorem line_inclination_45_implies_a_equals_1 (a : ℝ) : 
  (∃ x y : ℝ, a * x + (2 * a - 3) * y = 0 ∧ 
   Real.arctan ((3 - 2 * a) / a) = π / 4) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_implies_a_equals_1_l3657_365703


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3657_365772

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 72) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 36) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3657_365772


namespace NUMINAMATH_CALUDE_julians_debt_l3657_365769

/-- The amount Julian owes Jenny after borrowing additional money -/
def total_debt (initial_debt : ℕ) (borrowed_amount : ℕ) : ℕ :=
  initial_debt + borrowed_amount

/-- Theorem stating that Julian's total debt is 28 dollars -/
theorem julians_debt : total_debt 20 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_julians_debt_l3657_365769


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3657_365757

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 ∧ x ≠ -3 →
  (8 * x - 5) / (x^2 - 4 * x - 21) = (51 / 10) / (x - 7) + (29 / 10) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3657_365757


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3657_365797

theorem complex_equation_solution (z : ℂ) :
  z * (2 - 3*I) = 6 + 4*I → z = 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3657_365797


namespace NUMINAMATH_CALUDE_prob_A_given_B_l3657_365741

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (Ω.card : ℚ)

-- Define conditional probability
def conditional_prob (X Y : Finset Nat) : ℚ := P (X ∩ Y) / P Y

-- Theorem statement
theorem prob_A_given_B : conditional_prob A B = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_given_B_l3657_365741


namespace NUMINAMATH_CALUDE_tan_period_l3657_365701

/-- The period of y = tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l3657_365701


namespace NUMINAMATH_CALUDE_y_value_l3657_365706

theorem y_value : ∃ y : ℝ, y ≠ 0 ∧ y = 2 * (1 / y) * (-y) - 4 → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3657_365706


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l3657_365749

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l3657_365749


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_4410_l3657_365755

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_4410 :
  let factorization := prime_factorization 4410
  (factorization = [(2, 1), (3, 2), (5, 1), (7, 2)]) →
  (count_perfect_cube_factors 4410 = 1) := by sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_4410_l3657_365755


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l3657_365763

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_eq : a^2 + 2*b^2 = 3*c^2) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C ≥ Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l3657_365763


namespace NUMINAMATH_CALUDE_classroom_pencils_l3657_365704

/-- The number of pencils a teacher gives out to a classroom of students. -/
def pencils_given_out (num_students : ℕ) (dozens_per_student : ℕ) (pencils_per_dozen : ℕ) : ℕ :=
  num_students * dozens_per_student * pencils_per_dozen

/-- Theorem stating the total number of pencils given out in the classroom scenario. -/
theorem classroom_pencils : 
  pencils_given_out 96 7 12 = 8064 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l3657_365704


namespace NUMINAMATH_CALUDE_calculate_expression_l3657_365715

theorem calculate_expression : ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3657_365715


namespace NUMINAMATH_CALUDE_most_suitable_sampling_method_l3657_365795

-- Define the population structure
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

-- Define the sampling method
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | RemoveOneElderlyThenStratified

-- Define the suitability of a sampling method
def isMostSuitable (pop : Population) (sampleSize : ℕ) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.RemoveOneElderlyThenStratified

-- Theorem statement
theorem most_suitable_sampling_method 
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : ℕ)
  (h4 : sampleSize = 36) :
  isMostSuitable pop sampleSize SamplingMethod.RemoveOneElderlyThenStratified :=
by
  sorry

end NUMINAMATH_CALUDE_most_suitable_sampling_method_l3657_365795


namespace NUMINAMATH_CALUDE_function_properties_l3657_365744

-- Define the functions y₁ and y₂
def y₁ (a b x : ℝ) : ℝ := x^2 + a*x + b
def y₂ (x : ℝ) : ℝ := x^2 + x - 2

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, |y₁ a b x| ≤ |y₂ x|) →
  (a = 1 ∧ b = -2) ∧
  (∀ m : ℝ, (∀ x > 1, y₁ a b x > (m - 2)*x - m) → m < 2*Real.sqrt 2 + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3657_365744


namespace NUMINAMATH_CALUDE_bounded_diff_sequence_has_infinite_divisible_pairs_l3657_365780

/-- A sequence of positive integers with bounded differences -/
def BoundedDiffSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001

/-- The property of having infinitely many divisible pairs -/
def InfinitelyManyDivisiblePairs (a : ℕ → ℕ) : Prop :=
  ∀ k, ∃ p q, p > q ∧ q > k ∧ a q ∣ a p

/-- The main theorem -/
theorem bounded_diff_sequence_has_infinite_divisible_pairs
  (a : ℕ → ℕ) (h : BoundedDiffSequence a) :
  InfinitelyManyDivisiblePairs a :=
sorry

end NUMINAMATH_CALUDE_bounded_diff_sequence_has_infinite_divisible_pairs_l3657_365780


namespace NUMINAMATH_CALUDE_refrigerator_installation_cost_l3657_365739

theorem refrigerator_installation_cost 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (profit_percentage : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 12500)
  (h2 : discount_percentage = 0.20)
  (h3 : transport_cost = 125)
  (h4 : profit_percentage = 0.12)
  (h5 : selling_price = 17920) :
  ∃ (installation_cost : ℝ),
    installation_cost = 295 ∧
    selling_price = 
      (purchase_price / (1 - discount_percentage)) * 
      (1 + profit_percentage) + 
      transport_cost + 
      installation_cost :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_installation_cost_l3657_365739


namespace NUMINAMATH_CALUDE_mary_total_time_l3657_365719

-- Define the given conditions
def mac_download_time : ℕ := 10
def windows_download_time : ℕ := 3 * mac_download_time
def audio_glitch_time : ℕ := 2 * 4
def video_glitch_time : ℕ := 6
def glitch_time : ℕ := audio_glitch_time + video_glitch_time
def non_glitch_time : ℕ := 2 * glitch_time

-- Theorem statement
theorem mary_total_time :
  mac_download_time + windows_download_time + glitch_time + non_glitch_time = 82 :=
by sorry

end NUMINAMATH_CALUDE_mary_total_time_l3657_365719


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l3657_365784

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (A B C D : ℝ × ℝ)
  (is_cyclic : sorry)

/-- The area of a cyclic quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- The distance between two points in ℝ². -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area :
  ∀ (q : CyclicQuadrilateral),
    distance q.A q.B = 1 →
    distance q.B q.C = 3 →
    distance q.C q.D = 2 →
    distance q.D q.A = 2 →
    area q = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l3657_365784


namespace NUMINAMATH_CALUDE_fruit_store_theorem_l3657_365705

def fruit_problem (total_kg : ℕ) (total_cost : ℕ) 
                  (purchase_price_A purchase_price_B : ℕ)
                  (selling_price_A selling_price_B : ℕ) :=
  ∃ (kg_A kg_B : ℕ),
    -- Total kg constraint
    kg_A + kg_B = total_kg ∧ 
    -- Total cost constraint
    kg_A * purchase_price_A + kg_B * purchase_price_B = total_cost ∧
    -- Specific kg values
    kg_A = 65 ∧ kg_B = 75 ∧
    -- Profit calculation
    (kg_A * (selling_price_A - purchase_price_A) + 
     kg_B * (selling_price_B - purchase_price_B)) = 495

theorem fruit_store_theorem : 
  fruit_problem 140 1000 5 9 8 13 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_theorem_l3657_365705


namespace NUMINAMATH_CALUDE_sqrt_calculations_l3657_365787

theorem sqrt_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6) = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l3657_365787


namespace NUMINAMATH_CALUDE_circle_properties_l3657_365792

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

/-- Point of tangency -/
def point_of_tangency : ℝ × ℝ := (2, 0)

theorem circle_properties :
  (∃ (x y : ℝ), circle_equation x y ∧ x = 0 ∧ y = 0) ∧  -- Passes through origin
  (∀ (x y : ℝ), circle_equation x y → line_equation x y → (x, y) = point_of_tangency) ∧  -- Tangent at (2, 0)
  circle_equation (point_of_tangency.1) (point_of_tangency.2) :=  -- Point (2, 0) is on the circle
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3657_365792


namespace NUMINAMATH_CALUDE_negative_difference_l3657_365781

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l3657_365781


namespace NUMINAMATH_CALUDE_x_value_in_equation_l3657_365726

theorem x_value_in_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 71) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_equation_l3657_365726


namespace NUMINAMATH_CALUDE_min_coins_for_change_l3657_365791

/-- Represents the available denominations in cents -/
def denominations : List ℕ := [200, 100, 25, 10, 5, 1]

/-- Calculates the minimum number of bills and coins needed for change -/
def minCoins (amount : ℕ) : ℕ :=
  sorry

/-- The change amount in cents -/
def changeAmount : ℕ := 456

/-- Theorem stating that the minimum number of bills and coins for $4.56 change is 6 -/
theorem min_coins_for_change : minCoins changeAmount = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_change_l3657_365791


namespace NUMINAMATH_CALUDE_square_difference_equality_l3657_365782

theorem square_difference_equality : (25 + 15 + 8)^2 - (25^2 + 15^2 + 8^2) = 1390 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3657_365782


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l3657_365754

theorem partial_fraction_sum (P Q R : ℚ) : 
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 2 → 
    (x^2 + 5*x - 14) / ((x - 3)*(x + 1)*(x - 2)) = 
    P / (x - 3) + Q / (x + 1) + R / (x - 2)) →
  P + Q + R = 11.5 / 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l3657_365754


namespace NUMINAMATH_CALUDE_circle_triangle_area_l3657_365761

/-- Given a circle C with center (a, 2/a) that passes through the origin (0, 0)
    and intersects the x-axis at (2a, 0) and the y-axis at (0, 4/a),
    prove that the area of the triangle formed by these three points is 4. -/
theorem circle_triangle_area (a : ℝ) (ha : a ≠ 0) : 
  let center : ℝ × ℝ := (a, 2/a)
  let origin : ℝ × ℝ := (0, 0)
  let point_A : ℝ × ℝ := (2*a, 0)
  let point_B : ℝ × ℝ := (0, 4/a)
  let triangle_area := abs ((point_A.1 - origin.1) * (point_B.2 - origin.2)) / 2
  triangle_area = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l3657_365761


namespace NUMINAMATH_CALUDE_total_seashells_equation_l3657_365786

/-- The number of seashells Fred found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := 25

/-- The number of seashells Fred has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_equation : total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_equation_l3657_365786


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3657_365794

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

-- Define the conditions
def perpendicular (v w : Fin 2 → ℝ) : Prop := 
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop := 
  ∃ (k : ℝ), v = fun i ↦ k * (w i)

-- Theorem statement
theorem vector_sum_magnitude (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3657_365794


namespace NUMINAMATH_CALUDE_march_2020_production_theorem_l3657_365776

/-- Calculates the total toilet paper production for March 2020 after a production increase -/
def march_2020_toilet_paper_production (initial_production : ℕ) (increase_factor : ℕ) (days : ℕ) : ℕ :=
  (initial_production + initial_production * increase_factor) * days

/-- Theorem stating the total toilet paper production for March 2020 -/
theorem march_2020_production_theorem :
  march_2020_toilet_paper_production 7000 3 31 = 868000 := by
  sorry

#eval march_2020_toilet_paper_production 7000 3 31

end NUMINAMATH_CALUDE_march_2020_production_theorem_l3657_365776


namespace NUMINAMATH_CALUDE_skater_speeds_l3657_365778

theorem skater_speeds (V₁ V₂ : ℝ) (h1 : V₁ > 0) (h2 : V₂ > 0) 
  (h3 : (V₁ + V₂) / |V₁ - V₂| = 4) (h4 : V₁ = 6 ∨ V₂ = 6) :
  (V₁ = 10 ∧ V₂ = 6) ∨ (V₁ = 6 ∧ V₂ = 3.6) := by
  sorry

end NUMINAMATH_CALUDE_skater_speeds_l3657_365778


namespace NUMINAMATH_CALUDE_complex_multiplication_l3657_365774

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3657_365774


namespace NUMINAMATH_CALUDE_total_coins_remain_odd_cannot_achieve_equal_coins_l3657_365750

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- The initial state of Petya's coins -/
def initial_state : CoinState := { two_kopeck := 1, ten_kopeck := 0 }

/-- Represents a coin insertion operation -/
inductive InsertionOperation
  | insert_two_kopeck
  | insert_ten_kopeck

/-- Applies an insertion operation to a coin state -/
def apply_insertion (state : CoinState) (op : InsertionOperation) : CoinState :=
  match op with
  | InsertionOperation.insert_two_kopeck => 
      { two_kopeck := state.two_kopeck - 1, ten_kopeck := state.ten_kopeck + 5 }
  | InsertionOperation.insert_ten_kopeck => 
      { two_kopeck := state.two_kopeck + 5, ten_kopeck := state.ten_kopeck - 1 }

/-- The total number of coins in a given state -/
def total_coins (state : CoinState) : ℕ := state.two_kopeck + state.ten_kopeck

/-- Theorem stating that the total number of coins remains odd after any sequence of insertions -/
theorem total_coins_remain_odd (ops : List InsertionOperation) : 
  Odd (total_coins (ops.foldl apply_insertion initial_state)) := by
  sorry

/-- Theorem stating that Petya cannot achieve an equal number of two-kopeck and ten-kopeck coins -/
theorem cannot_achieve_equal_coins (ops : List InsertionOperation) : 
  let final_state := ops.foldl apply_insertion initial_state
  ¬(final_state.two_kopeck = final_state.ten_kopeck) := by
  sorry

end NUMINAMATH_CALUDE_total_coins_remain_odd_cannot_achieve_equal_coins_l3657_365750


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l3657_365775

/-- The line equation y = (5/3)x - 15 -/
def line_equation (x y : ℝ) : Prop := y = (5/3) * x - 15

/-- Point P is where the line crosses the x-axis -/
def point_P (x : ℝ) : Prop := line_equation x 0

/-- Point Q is where the line crosses the y-axis -/
def point_Q (y : ℝ) : Prop := line_equation 0 y

/-- Point T(r, s) is on the line -/
def point_T (r s : ℝ) : Prop := line_equation r s

/-- T is between P and Q on the line segment -/
def T_between_P_Q (r s : ℝ) : Prop := 
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧ 
  ((0 ≤ r ∧ r ≤ px) ∨ (px ≤ r ∧ r ≤ 0)) ∧
  ((qy ≤ s ∧ s ≤ 0) ∨ (0 ≤ s ∧ s ≤ qy))

/-- Area of triangle POQ is twice the area of triangle TOQ -/
def area_condition (r s : ℝ) : Prop :=
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧
  (1/2 * px * abs qy) = 2 * (1/2 * px * abs (s - qy))

theorem line_segment_point_sum : 
  ∀ (r s : ℝ), point_T r s ∧ T_between_P_Q r s ∧ area_condition r s → r + s = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l3657_365775


namespace NUMINAMATH_CALUDE_max_binomial_coeff_expansion_l3657_365771

theorem max_binomial_coeff_expansion (m : ℕ) : 
  (∀ x : ℝ, x > 0 → (5 / Real.sqrt x - x)^m = 256) → 
  (∃ k : ℕ, k ≤ m ∧ Nat.choose m k = 6 ∧ ∀ j : ℕ, j ≤ m → Nat.choose m j ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_binomial_coeff_expansion_l3657_365771


namespace NUMINAMATH_CALUDE_james_alice_equation_equivalence_l3657_365738

theorem james_alice_equation_equivalence (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) ↔ (d = -16 ∧ e = 55) := by
  sorry

end NUMINAMATH_CALUDE_james_alice_equation_equivalence_l3657_365738


namespace NUMINAMATH_CALUDE_trig_identity_second_quadrant_l3657_365727

theorem trig_identity_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by sorry

end NUMINAMATH_CALUDE_trig_identity_second_quadrant_l3657_365727


namespace NUMINAMATH_CALUDE_slope_constraint_implies_a_bound_l3657_365747

/-- Given a function f(x) = x ln(x) + ax^2, if there exists a point where the slope is 3,
    then a is greater than or equal to -1 / (2e^3). -/
theorem slope_constraint_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (Real.log x + 1 + 2 * a * x = 3)) →
  a ≥ -1 / (2 * Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_slope_constraint_implies_a_bound_l3657_365747


namespace NUMINAMATH_CALUDE_largest_number_l3657_365730

theorem largest_number (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : -1 < b ∧ b < 0) :
  (a - b) = max a (max (a * b) (max (a - b) (a + b))) := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3657_365730


namespace NUMINAMATH_CALUDE_handshake_problem_l3657_365743

/-- The number of handshakes in a group of n people where each person shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of men in the group -/
def num_men : ℕ := 60

theorem handshake_problem :
  handshakes num_men = 1770 :=
sorry

#eval handshakes num_men

end NUMINAMATH_CALUDE_handshake_problem_l3657_365743


namespace NUMINAMATH_CALUDE_olivers_mom_money_l3657_365720

/-- Calculates the amount of money Oliver's mom gave him -/
theorem olivers_mom_money (initial : ℕ) (spent : ℕ) (final : ℕ) : 
  initial - spent + (final - (initial - spent)) = final ∧ 
  final - (initial - spent) = 32 :=
by
  sorry

#check olivers_mom_money 33 4 61

end NUMINAMATH_CALUDE_olivers_mom_money_l3657_365720


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l3657_365708

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) :
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 5 →
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = 5 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l3657_365708


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3657_365700

theorem difference_of_squares_special_case : (733 : ℤ) * 733 - 732 * 734 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3657_365700


namespace NUMINAMATH_CALUDE_perimeter_unchanged_after_adding_tiles_l3657_365716

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

/-- The theorem stating that adding two tiles can maintain the same perimeter -/
theorem perimeter_unchanged_after_adding_tiles :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 9 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 2 ∧
    final.perimeter = 16 :=
  sorry

end NUMINAMATH_CALUDE_perimeter_unchanged_after_adding_tiles_l3657_365716


namespace NUMINAMATH_CALUDE_root_product_squared_plus_one_l3657_365793

theorem root_product_squared_plus_one (a b c : ℂ) : 
  (a^3 + 20*a^2 + a + 5 = 0) →
  (b^3 + 20*b^2 + b + 5 = 0) →
  (c^3 + 20*c^2 + c + 5 = 0) →
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) = 229 := by
  sorry

end NUMINAMATH_CALUDE_root_product_squared_plus_one_l3657_365793


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3657_365766

theorem complex_equation_proof (a b : ℝ) : 
  (a + b * Complex.I) / (2 - Complex.I) = (3 : ℂ) + Complex.I → a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3657_365766


namespace NUMINAMATH_CALUDE_distance_to_town_l3657_365753

theorem distance_to_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_town_l3657_365753


namespace NUMINAMATH_CALUDE_smallest_n_purple_candy_l3657_365734

def orange_candy : ℕ := 10
def yellow_candy : ℕ := 16
def gray_candy : ℕ := 18
def purple_candy_cost : ℕ := 18

theorem smallest_n_purple_candy : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (total_cost : ℕ), 
    total_cost = orange_candy * n ∧
    total_cost = yellow_candy * n ∧
    total_cost = gray_candy * n ∧
    total_cost = purple_candy_cost * n) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (total_cost : ℕ), 
      total_cost = orange_candy * m ∧
      total_cost = yellow_candy * m ∧
      total_cost = gray_candy * m ∧
      total_cost = purple_candy_cost * m)) ∧
  n = 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_purple_candy_l3657_365734


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l3657_365762

theorem gcd_of_squares_sum : Nat.gcd (122^2 + 234^2 + 344^2) (123^2 + 235^2 + 343^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l3657_365762


namespace NUMINAMATH_CALUDE_green_light_most_probable_l3657_365709

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle duration
def total_duration : ℕ := red_duration + yellow_duration + green_duration

-- Define the probabilities of encountering each light
def prob_red : ℚ := red_duration / total_duration
def prob_yellow : ℚ := yellow_duration / total_duration
def prob_green : ℚ := green_duration / total_duration

-- Theorem: The probability of encountering a green light is higher than the other lights
theorem green_light_most_probable : 
  prob_green > prob_red ∧ prob_green > prob_yellow :=
sorry

end NUMINAMATH_CALUDE_green_light_most_probable_l3657_365709


namespace NUMINAMATH_CALUDE_max_value_p_l3657_365724

theorem max_value_p (p q r s t u v w : ℕ+) : 
  (p + q + r + s = 35) →
  (q + r + s + t = 35) →
  (r + s + t + u = 35) →
  (s + t + u + v = 35) →
  (t + u + v + w = 35) →
  (q + v = 14) →
  (∀ x : ℕ+, x ≤ p → 
    ∃ q' r' s' t' u' v' w' : ℕ+,
      (x + q' + r' + s' = 35) ∧
      (q' + r' + s' + t' = 35) ∧
      (r' + s' + t' + u' = 35) ∧
      (s' + t' + u' + v' = 35) ∧
      (t' + u' + v' + w' = 35) ∧
      (q' + v' = 14)) →
  p = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_p_l3657_365724


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3657_365756

-- Define f as a differentiable function on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (hf_diff : Differentiable ℝ f)
variable (hf_domain : ∀ x, x > 0 → f x ≠ 0)

-- Define the condition f(x) > x * f'(x)
variable (hf_cond : ∀ x, x > 0 → f x > x * (deriv f x))

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem inequality_solution_set :
  ∀ x > 0, x^2 * f (1/x) - f x < 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3657_365756


namespace NUMINAMATH_CALUDE_constant_function_theorem_l3657_365721

theorem constant_function_theorem (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y)) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
by sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l3657_365721


namespace NUMINAMATH_CALUDE_rectangle_segment_comparison_l3657_365748

/-- Given a rectangle ABCD with specific properties, prove AM > BK -/
theorem rectangle_segment_comparison (A B C D M K : ℝ × ℝ) : 
  let AB : ℝ := 2
  let BD : ℝ := Real.sqrt 7
  let AC : ℝ := Real.sqrt (AB^2 + BD^2 - AB^2)
  -- Rectangle properties
  (B.1 - A.1 = AB ∧ B.2 = A.2) →
  (C.1 = B.1 ∧ C.2 - A.2 = AC) →
  (D.1 = A.1 ∧ D.2 = C.2) →
  -- M divides CD in 1:2 ratio
  (M.1 - C.1 = (1/3) * (D.1 - C.1) ∧ M.2 - C.2 = (1/3) * (D.2 - C.2)) →
  -- K is midpoint of AD
  (K.1 = (A.1 + D.1) / 2 ∧ K.2 = (A.2 + D.2) / 2) →
  -- Prove AM > BK
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) > Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_segment_comparison_l3657_365748


namespace NUMINAMATH_CALUDE_star_perimeter_is_160_l3657_365725

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 5

/-- The side length of the square in cm -/
def square_side_length : ℝ := 4 * circle_radius

/-- The number of sides in the star -/
def star_sides : ℕ := 8

/-- The perimeter of the star in cm -/
def star_perimeter : ℝ := star_sides * square_side_length

/-- Theorem stating that the perimeter of the star is 160 cm -/
theorem star_perimeter_is_160 : star_perimeter = 160 := by
  sorry

end NUMINAMATH_CALUDE_star_perimeter_is_160_l3657_365725


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l3657_365723

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-1) = 0 ∧
  ∀ x, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2

/-- The theorem stating that the quadratic function satisfying the given conditions
    must be f(x) = 1/4(x+1)^2 -/
theorem quadratic_function_unique :
  ∀ f : ℝ → ℝ, QuadraticFunction f → ∀ x, f x = (1/4) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l3657_365723


namespace NUMINAMATH_CALUDE_common_chord_length_l3657_365770

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by
  sorry

#check common_chord_length

end NUMINAMATH_CALUDE_common_chord_length_l3657_365770


namespace NUMINAMATH_CALUDE_solve_salary_problem_l3657_365752

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 17000

theorem solve_salary_problem : 
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 170000 := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l3657_365752


namespace NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l3657_365707

/-- Given a point P(3, 4) on the terminal side of angle α, prove that sin α + cos α = 8/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : Real × Real := (3, 4)
  (P.1 = 3 ∧ P.2 = 4) →  -- Point P has coordinates (3, 4)
  (P.1^2 + P.2^2 = 5^2) →  -- P is on the unit circle with radius 5
  (Real.sin α = P.2 / 5 ∧ Real.cos α = P.1 / 5) →  -- Definition of sin and cos for this point
  Real.sin α + Real.cos α = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l3657_365707


namespace NUMINAMATH_CALUDE_door_challenge_sequences_l3657_365764

/-- Represents the number of doors and family members -/
def n : ℕ := 7

/-- Represents the number of binary choices made after the first person -/
def m : ℕ := n - 1

/-- The number of possible sequences given n doors and m binary choices -/
def num_sequences (n m : ℕ) : ℕ := 2^m

theorem door_challenge_sequences :
  n = 7 → m = 6 → num_sequences n m = 64 := by
  sorry

end NUMINAMATH_CALUDE_door_challenge_sequences_l3657_365764


namespace NUMINAMATH_CALUDE_total_spent_on_cards_l3657_365711

def digimon_pack_price : ℚ := 4.45
def digimon_pack_count : ℕ := 4
def baseball_deck_price : ℚ := 6.06

theorem total_spent_on_cards :
  digimon_pack_price * digimon_pack_count + baseball_deck_price = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_cards_l3657_365711


namespace NUMINAMATH_CALUDE_volume_of_cut_cube_piece_l3657_365751

theorem volume_of_cut_cube_piece (cube_edge : ℝ) (piece_base_side : ℝ) (piece_height : ℝ) : 
  cube_edge = 1 →
  piece_base_side = 1/3 →
  piece_height = 1/3 →
  (1/3) * (piece_base_side^2) * piece_height = 1/81 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_cut_cube_piece_l3657_365751


namespace NUMINAMATH_CALUDE_remainder_after_adding_2947_l3657_365759

theorem remainder_after_adding_2947 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2947_l3657_365759


namespace NUMINAMATH_CALUDE_latest_departure_time_l3657_365742

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Represents the flight constraints -/
structure FlightConstraints where
  flightDepartureTime : Time
  checkInTime : Nat
  driveTime : Nat
  parkAndWalkTime : Nat

theorem latest_departure_time (constraints : FlightConstraints) 
  (h1 : constraints.flightDepartureTime = ⟨20, 0⟩)
  (h2 : constraints.checkInTime = 120)
  (h3 : constraints.driveTime = 45)
  (h4 : constraints.parkAndWalkTime = 15) :
  let latestDepartureTime := ⟨17, 0⟩
  timeDifference constraints.flightDepartureTime latestDepartureTime = 
    constraints.checkInTime + constraints.driveTime + constraints.parkAndWalkTime :=
by sorry

end NUMINAMATH_CALUDE_latest_departure_time_l3657_365742


namespace NUMINAMATH_CALUDE_game_problem_l3657_365731

/-- Represents the game setup -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)
  (prob_threshold : ℚ)

/-- Calculates the minimum number of boxes to eliminate -/
def min_boxes_to_eliminate (setup : GameSetup) : ℕ :=
  setup.total_boxes - 2 * setup.valuable_boxes

/-- Theorem statement for the game problem -/
theorem game_problem (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 5)
  (h3 : setup.prob_threshold = 1/2) :
  min_boxes_to_eliminate setup = 20 := by
  sorry

#eval min_boxes_to_eliminate { total_boxes := 30, valuable_boxes := 5, prob_threshold := 1/2 }

end NUMINAMATH_CALUDE_game_problem_l3657_365731


namespace NUMINAMATH_CALUDE_jakobs_class_size_l3657_365737

theorem jakobs_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 200 ∧
    b % 4 = 2 ∧ b % 5 = 2 ∧ b % 6 = 2 ∧
    b = 122 := by sorry

end NUMINAMATH_CALUDE_jakobs_class_size_l3657_365737


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l3657_365710

theorem smallest_perfect_square_divisible_by_4_and_5 :
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 4 = 0 → n % 5 = 0 → n ≥ 400 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l3657_365710


namespace NUMINAMATH_CALUDE_simplify_expression_l3657_365729

theorem simplify_expression (w : ℝ) : (5 - 2*w) - (4 + 5*w) = 1 - 7*w := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3657_365729


namespace NUMINAMATH_CALUDE_rocket_coaster_total_cars_l3657_365779

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ
  total_capacity : ℕ

/-- The Rocket Coaster satisfies the given conditions -/
def rocket_coaster : RollerCoaster :=
  { four_passenger_cars := 9,
    six_passenger_cars := 6,
    total_capacity := 72 }

/-- The total number of cars on the Rocket Coaster -/
def total_cars (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars + rc.six_passenger_cars

/-- Theorem stating that the total number of cars on the Rocket Coaster is 15 -/
theorem rocket_coaster_total_cars :
  total_cars rocket_coaster = 15 ∧
  4 * rocket_coaster.four_passenger_cars + 6 * rocket_coaster.six_passenger_cars = rocket_coaster.total_capacity :=
by sorry

#eval total_cars rocket_coaster -- Should output 15

end NUMINAMATH_CALUDE_rocket_coaster_total_cars_l3657_365779


namespace NUMINAMATH_CALUDE_f_zero_and_range_l3657_365712

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

-- State the theorem
theorem f_zero_and_range :
  -- f(x) has one zero in (-1, 1)
  ∃ (x : ℝ), -1 < x ∧ x < 1 ∧ f a x = 0 →
  -- The range of a
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a ∧ a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) ∧
  -- When a = 32/17, the solution is 1/2
  (a = 32/17 → f (32/17) (1/2) = 0) :=
sorry


end NUMINAMATH_CALUDE_f_zero_and_range_l3657_365712


namespace NUMINAMATH_CALUDE_second_quadrant_m_range_l3657_365732

theorem second_quadrant_m_range (m : ℝ) : 
  (m^2 - 1 < 0 ∧ m > 0) → (0 < m ∧ m < 1) := by sorry

end NUMINAMATH_CALUDE_second_quadrant_m_range_l3657_365732


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3657_365758

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the slope of one of its asymptotes is 2, then its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3657_365758


namespace NUMINAMATH_CALUDE_star_properties_l3657_365767

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 1) - 3

-- Theorem statement
theorem star_properties :
  (¬ ∀ x y : ℝ, star x y = star y x) ∧ 
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧ 
  (star 0 1 = 1) := by
  sorry


end NUMINAMATH_CALUDE_star_properties_l3657_365767


namespace NUMINAMATH_CALUDE_fraction_comparisons_l3657_365783

theorem fraction_comparisons :
  ∀ (a b c : ℚ),
  (0 < b) → (b < 1) → (a * b < a) ∧
  (0 < c) → (c < b) → (a * b > a * c) ∧
  (0 < b) → (b < 1) → (a < a / b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparisons_l3657_365783


namespace NUMINAMATH_CALUDE_initially_calculated_average_height_l3657_365789

theorem initially_calculated_average_height
  (n : ℕ)
  (wrong_height actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 183) :
  let initially_calculated_average := 
    (n * actual_average - (wrong_height - actual_height)) / n
  initially_calculated_average = 181 := by
sorry

end NUMINAMATH_CALUDE_initially_calculated_average_height_l3657_365789


namespace NUMINAMATH_CALUDE_deductive_reasoning_properties_l3657_365785

-- Define the properties of deductive reasoning
def is_general_to_specific (r : Type) : Prop := sorry
def conclusion_always_correct (r : Type) : Prop := sorry
def has_syllogism_form (r : Type) : Prop := sorry
def correctness_depends_on_premises_and_form (r : Type) : Prop := sorry

-- Define deductive reasoning
def deductive_reasoning : Type := sorry

-- Theorem stating that exactly 3 out of 4 statements are correct
theorem deductive_reasoning_properties :
  is_general_to_specific deductive_reasoning ∧
  ¬conclusion_always_correct deductive_reasoning ∧
  has_syllogism_form deductive_reasoning ∧
  correctness_depends_on_premises_and_form deductive_reasoning :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_properties_l3657_365785


namespace NUMINAMATH_CALUDE_inequality_proof_l3657_365798

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3657_365798


namespace NUMINAMATH_CALUDE_smallest_sum_with_gcd_lcm_condition_l3657_365733

theorem smallest_sum_with_gcd_lcm_condition (a b : ℕ+) : 
  (Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) → 
  (∀ c d : ℕ+, (Nat.gcd c d + Nat.lcm c d = 3 * (c + d)) → (a + b ≤ c + d)) → 
  a + b = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_gcd_lcm_condition_l3657_365733


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l3657_365773

/-- The amount made per t-shirt, given the number of t-shirts sold and the total amount made from t-shirts. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $62. -/
theorem razorback_tshirt_profit : amount_per_tshirt 183 11346 = 62 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l3657_365773


namespace NUMINAMATH_CALUDE_maximum_marks_l3657_365765

theorem maximum_marks : ∃ M : ℕ, 
  (M ≥ 434) ∧ 
  (M < 435) ∧ 
  (⌈(0.45 : ℝ) * (M : ℝ)⌉ = 130 + 65) := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_l3657_365765
