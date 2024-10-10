import Mathlib

namespace similar_triangle_leg_length_l1485_148543

theorem similar_triangle_leg_length 
  (a b c d : ℝ) 
  (h1 : a^2 + 24^2 = 25^2) 
  (h2 : b^2 + c^2 = d^2) 
  (h3 : d / 25 = 100 / 25) 
  (h4 : b / a = d / 25) 
  (h5 : c / 24 = d / 25) : 
  c = 28 := by
sorry

end similar_triangle_leg_length_l1485_148543


namespace shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l1485_148566

/-- The area of the shaded region between a circle circumscribing two externally tangent circles with radii 3 and 5 -/
theorem shaded_area_of_circumscribed_circles (π : ℝ) : ℝ := by
  -- Define the radii of the two smaller circles
  let r₁ : ℝ := 3
  let r₂ : ℝ := 5

  -- Define the radius of the larger circumscribing circle
  let R : ℝ := r₁ + r₂ + r₂

  -- Define the areas of the circles
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := π * r₂^2
  let A_large : ℝ := π * R^2

  -- Define the shaded area
  let shaded_area : ℝ := A_large - A₁ - A₂

  -- Prove that the shaded area equals 135π
  sorry

/-- The main theorem stating that the shaded area is equal to 135π -/
theorem shaded_area_equals_135π (π : ℝ) : shaded_area_of_circumscribed_circles π = 135 * π := by
  sorry

end shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l1485_148566


namespace smallest_prime_divisor_of_sum_l1485_148502

theorem smallest_prime_divisor_of_sum (p : Nat) 
  (h1 : Prime 7) (h2 : Prime 11) : 
  (p.Prime ∧ p ∣ (7^13 + 11^15) ∧ ∀ q, q.Prime → q ∣ (7^13 + 11^15) → p ≤ q) → p = 2 := by
  sorry

end smallest_prime_divisor_of_sum_l1485_148502


namespace intersection_A_complement_B_l1485_148523

-- Define the set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 3}

-- Define the set B
def B : Set ℤ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ Bᶜ = {0, 3} := by
  sorry

end intersection_A_complement_B_l1485_148523


namespace subset_condition_l1485_148587

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x - 4) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 5) > 0}

-- State the theorem
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4 ≤ a ∧ a < 5 := by sorry

end subset_condition_l1485_148587


namespace inequality_holds_iff_a_in_range_l1485_148546

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔ 
  (a ≤ -2 ∨ a ≥ 1) := by
  sorry

end inequality_holds_iff_a_in_range_l1485_148546


namespace bottles_poured_is_four_l1485_148599

def cylinder_capacity : ℚ := 80

def initial_fullness : ℚ := 3/4

def final_fullness : ℚ := 4/5

def bottles_poured (capacity : ℚ) (initial : ℚ) (final : ℚ) : ℚ :=
  capacity * final - capacity * initial

theorem bottles_poured_is_four :
  bottles_poured cylinder_capacity initial_fullness final_fullness = 4 := by
  sorry

end bottles_poured_is_four_l1485_148599


namespace ball_338_in_cup_360_l1485_148575

/-- The number of cups in the circle. -/
def n : ℕ := 1000

/-- The step size for placing balls. -/
def step : ℕ := 7

/-- The index of the ball we're interested in. -/
def ball_index : ℕ := 338

/-- Function to calculate the cup number for a given ball index. -/
def cup_number (k : ℕ) : ℕ :=
  (1 + step * (k - 1)) % n

theorem ball_338_in_cup_360 : cup_number ball_index = 360 := by
  sorry

end ball_338_in_cup_360_l1485_148575


namespace smallest_sum_A_plus_b_l1485_148570

theorem smallest_sum_A_plus_b : 
  ∀ (A : ℕ) (b : ℕ),
    A < 4 →
    A > 0 →
    b > 5 →
    21 * A = 3 * b + 3 →
    ∀ (A' : ℕ) (b' : ℕ),
      A' < 4 →
      A' > 0 →
      b' > 5 →
      21 * A' = 3 * b' + 3 →
      A + b ≤ A' + b' :=
by sorry

end smallest_sum_A_plus_b_l1485_148570


namespace loan_division_l1485_148514

/-- Given a total sum of 2730 divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is 1680. -/
theorem loan_division (total : ℝ) (part1 part2 : ℝ) : 
  total = 2730 →
  part1 + part2 = total →
  (part1 * 3 * 8) / 100 = (part2 * 5 * 3) / 100 →
  part2 = 1680 := by
  sorry

end loan_division_l1485_148514


namespace chris_savings_l1485_148535

/-- Chris's savings problem -/
theorem chris_savings (total : ℕ) (grandmother : ℕ) (parents : ℕ) (aunt_uncle : ℕ) 
  (h1 : total = 279)
  (h2 : grandmother = 25)
  (h3 : parents = 75)
  (h4 : aunt_uncle = 20) :
  total - (grandmother + parents + aunt_uncle) = 159 := by
  sorry

end chris_savings_l1485_148535


namespace simplify_expression_combine_like_terms_l1485_148528

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem simplify_expression :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b :=
by sorry

-- Theorem 2
theorem combine_like_terms :
  3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 :=
by sorry

end simplify_expression_combine_like_terms_l1485_148528


namespace stream_speed_l1485_148586

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 18 →
  downstream_distance = 48 →
  upstream_distance = 32 →
  ∃ (time : ℝ), time > 0 ∧
    time * (boat_speed + 3.6) = downstream_distance ∧
    time * (boat_speed - 3.6) = upstream_distance :=
by sorry

end stream_speed_l1485_148586


namespace horner_method_correctness_l1485_148527

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  let v3 := v2 * x + 3
  let v4 := v3 * x + 1
  v4 * x + 1

theorem horner_method_correctness : f 3 = horner_eval 3 := by sorry

end horner_method_correctness_l1485_148527


namespace range_of_b_l1485_148520

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem range_of_b (a : ℝ) (h : a = -3) :
  (∀ b : ℝ, (A ∩ B a b).Nonempty) → ∀ b : ℝ, b > -1 :=
by sorry

end range_of_b_l1485_148520


namespace unique_three_digit_number_l1485_148518

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Checks if the digits of a ThreeDigitNumber are distinct -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- The main theorem stating that 156 is the only number satisfying the conditions -/
theorem unique_three_digit_number : 
  ∀ n : ThreeDigitNumber, 
    has_distinct_digits n → 
    (100 ≤ to_nat n) ∧ (to_nat n ≤ 999) → 
    (to_nat n = (n.1 + n.2.1 + n.2.2) * (n.1 + n.2.1 + n.2.2 + 1)) → 
    n = (1, 5, 6) :=
by sorry

end unique_three_digit_number_l1485_148518


namespace closest_to_half_at_seven_dips_l1485_148568

/-- The number of unit cubes --/
def num_cubes : ℕ := 1729

/-- The number of faces per cube --/
def faces_per_cube : ℕ := 6

/-- The total number of faces --/
def total_faces : ℕ := num_cubes * faces_per_cube

/-- The expected number of painted faces per dip --/
def painted_per_dip : ℚ := 978

/-- The recurrence relation for painted faces --/
def painted_faces (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => painted_faces n * (1566 / 1729) + painted_per_dip

/-- The theorem to prove --/
theorem closest_to_half_at_seven_dips :
  ∀ k : ℕ, k ≠ 7 →
  |painted_faces 7 - (total_faces / 2)| < |painted_faces k - (total_faces / 2)| :=
sorry

end closest_to_half_at_seven_dips_l1485_148568


namespace ellipse_iff_k_in_range_l1485_148571

/-- The equation of an ellipse in the form (x^2 / (3+k)) + (y^2 / (2-k)) = 1 -/
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
def k_range : Set ℝ :=
  {k | k ∈ (Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) 2)}

/-- Theorem stating that the equation represents an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_in_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ k_range :=
sorry

end ellipse_iff_k_in_range_l1485_148571


namespace boat_downstream_speed_l1485_148547

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (b : BoatSpeed) : ℝ :=
  2 * b.stillWater - b.upstream

/-- Theorem stating that a boat with 11 km/hr speed in still water and 7 km/hr upstream 
    will have a downstream speed of 15 km/hr -/
theorem boat_downstream_speed :
  let b : BoatSpeed := { stillWater := 11, upstream := 7 }
  downstreamSpeed b = 15 := by sorry

end boat_downstream_speed_l1485_148547


namespace intersection_point_property_l1485_148548

/-- The x-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def a : ℝ := by sorry

/-- The y-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def b : ℝ := by sorry

/-- The intersection point satisfies the equation of y = 1/x -/
axiom inverse_prop : b = 1 / a

/-- The intersection point satisfies the equation of y = x + 2 -/
axiom linear : b = a + 2

theorem intersection_point_property : a - a * b - b = -3 := by sorry

end intersection_point_property_l1485_148548


namespace cannot_form_right_triangle_l1485_148567

theorem cannot_form_right_triangle : ¬ (9^2 + 16^2 = 25^2) := by
  sorry

end cannot_form_right_triangle_l1485_148567


namespace complex_magnitude_equals_five_l1485_148515

theorem complex_magnitude_equals_five (t : ℝ) (ht : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by
sorry

end complex_magnitude_equals_five_l1485_148515


namespace least_difference_consecutive_primes_l1485_148510

theorem least_difference_consecutive_primes (x y z p : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧  -- x < y < z
  y - x > 5 ∧  -- y - x > 5
  Even x ∧  -- x is an even integer
  Odd y ∧ Odd z ∧  -- y and z are odd integers
  (∃ k : ℕ, y^2 + x^2 = k * p) ∧  -- (y^2 + x^2) is divisible by a specific prime p
  Prime p →  -- p is prime
  (∃ s : ℕ, s = z - x ∧ ∀ t : ℕ, t = z - x → s ≤ t) ∧ s = 11  -- The least possible value s of z - x is 11
  := by sorry

end least_difference_consecutive_primes_l1485_148510


namespace least_common_multiple_18_35_l1485_148598

theorem least_common_multiple_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end least_common_multiple_18_35_l1485_148598


namespace graduating_class_size_l1485_148592

/-- Given a graduating class where there are 208 boys and 69 more girls than boys,
    prove that the total number of students is 485. -/
theorem graduating_class_size :
  ∀ (boys girls total : ℕ),
  boys = 208 →
  girls = boys + 69 →
  total = boys + girls →
  total = 485 := by
sorry

end graduating_class_size_l1485_148592


namespace product_of_sums_l1485_148529

theorem product_of_sums (a b c d : ℚ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : (a + c) * (a + d) = 1)
  (h2 : (b + c) * (b + d) = 1) :
  (a + c) * (b + c) = -1 := by
sorry

end product_of_sums_l1485_148529


namespace min_triangles_proof_l1485_148537

/-- Represents an 8x8 square with one corner cell removed -/
structure ModifiedSquare where
  side_length : ℕ
  removed_cell_area : ℕ
  total_area : ℕ

/-- Represents a triangulation of the modified square -/
structure Triangulation where
  num_triangles : ℕ
  triangle_area : ℝ

/-- The minimum number of equal-area triangles that can divide the modified square -/
def min_triangles : ℕ := 18

theorem min_triangles_proof (s : ModifiedSquare) (t : Triangulation) :
  s.side_length = 8 ∧ 
  s.removed_cell_area = 1 ∧ 
  s.total_area = s.side_length * s.side_length - s.removed_cell_area ∧
  t.triangle_area = s.total_area / t.num_triangles ∧
  t.triangle_area ≤ 3.5 →
  t.num_triangles ≥ min_triangles :=
sorry

end min_triangles_proof_l1485_148537


namespace priya_speed_calculation_l1485_148593

/-- Priya's speed in km/h -/
def priya_speed : ℝ := 30

/-- Riya's speed in km/h -/
def riya_speed : ℝ := 20

/-- Time traveled in hours -/
def time : ℝ := 0.5

/-- Distance between Riya and Priya after traveling -/
def distance : ℝ := 25

theorem priya_speed_calculation :
  (riya_speed + priya_speed) * time = distance :=
by sorry

end priya_speed_calculation_l1485_148593


namespace perfect_square_condition_l1485_148555

theorem perfect_square_condition (k : ℝ) :
  (∀ x y : ℝ, ∃ z : ℝ, 4 * x^2 - (k - 1) * x * y + 9 * y^2 = z^2) →
  k = 13 ∨ k = -11 := by
  sorry

end perfect_square_condition_l1485_148555


namespace base4_sum_equals_2133_l1485_148524

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Adds two base 4 numbers --/
def base4_add (a b : Base4) : Base4 := sorry

theorem base4_sum_equals_2133 :
  let a := to_base4 2
  let b := to_base4 (4 + 3)
  let c := to_base4 (16 + 12 + 2)
  let d := to_base4 (256 + 192 + 0)
  base4_add (base4_add (base4_add a b) c) d = to_base4 (512 + 48 + 12 + 3) := by
  sorry

end base4_sum_equals_2133_l1485_148524


namespace tan_150_degrees_l1485_148579

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by sorry

end tan_150_degrees_l1485_148579


namespace zero_smallest_natural_l1485_148507

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end zero_smallest_natural_l1485_148507


namespace meters_to_cm_conversion_l1485_148511

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Proves that 3.5 meters is equal to 350 centimeters -/
theorem meters_to_cm_conversion : 3.5 * meters_to_cm = 350 := by
  sorry

end meters_to_cm_conversion_l1485_148511


namespace six_balls_three_boxes_l1485_148595

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 92 := by
  sorry

end six_balls_three_boxes_l1485_148595


namespace distribute_equals_choose_l1485_148534

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_equals_choose :
  distribute 10 7 = choose 9 6 := by sorry

end distribute_equals_choose_l1485_148534


namespace proposition_and_converse_l1485_148590

theorem proposition_and_converse : 
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  ¬(∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
by sorry

end proposition_and_converse_l1485_148590


namespace triangle_max_area_l1485_148532

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos C - c / 2 = b →
  a = 2 * Real.sqrt 3 →
  (∃ (S : ℝ), S = (1 / 2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1 / 2) * b * c * Real.sin A → S' ≤ Real.sqrt 3) :=
by sorry

end triangle_max_area_l1485_148532


namespace depth_difference_is_four_l1485_148503

/-- The depth of Mark's pond in feet -/
def marks_pond_depth : ℕ := 19

/-- The depth of Peter's pond in feet -/
def peters_pond_depth : ℕ := 5

/-- The difference between Mark's pond depth and 3 times Peter's pond depth -/
def depth_difference : ℕ := marks_pond_depth - 3 * peters_pond_depth

theorem depth_difference_is_four :
  depth_difference = 4 :=
by sorry

end depth_difference_is_four_l1485_148503


namespace complex_power_210_deg_60_l1485_148539

theorem complex_power_210_deg_60 :
  (Complex.exp (210 * π / 180 * I)) ^ 60 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end complex_power_210_deg_60_l1485_148539


namespace village_population_l1485_148557

theorem village_population (population : ℕ) : 
  (60 : ℕ) * population = 23040 * 100 → population = 38400 := by
  sorry

end village_population_l1485_148557


namespace angle_A_in_triangle_l1485_148564

-- Define the triangle ABC
structure Triangle where
  A : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem angle_A_in_triangle (abc : Triangle) (h1 : abc.b = 8) (h2 : abc.c = 8 * Real.sqrt 3) 
  (h3 : abc.S = 16 * Real.sqrt 3) : 
  abc.A = π / 6 ∨ abc.A = 5 * π / 6 := by
  sorry

end angle_A_in_triangle_l1485_148564


namespace min_triangle_area_l1485_148556

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (p : ℤ)
  (q : ℤ)

/-- Calculate the area of the triangle using the Shoelace formula -/
def triangleArea (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.q - 18 * t.p|

/-- Theorem: The minimum area of triangle ABC is 3 -/
theorem min_triangle_area :
  ∃ (t : Triangle), ∀ (t' : Triangle), triangleArea t ≤ triangleArea t' ∧ triangleArea t = 3 :=
sorry

end min_triangle_area_l1485_148556


namespace kozlov_inequality_l1485_148569

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end kozlov_inequality_l1485_148569


namespace cut_pyramid_volume_l1485_148530

/-- The volume of a smaller pyramid cut from a right square pyramid -/
theorem cut_pyramid_volume (base_edge original_height slant_edge cut_height : ℝ) : 
  base_edge = 12 * Real.sqrt 2 →
  slant_edge = 15 →
  original_height = Real.sqrt (slant_edge^2 - (base_edge/2)^2) →
  cut_height = 5 →
  cut_height < original_height →
  (1/3) * (base_edge * (original_height - cut_height) / original_height)^2 * (original_height - cut_height) = 2048/27 :=
by sorry

end cut_pyramid_volume_l1485_148530


namespace alex_singing_probability_alex_singing_probability_proof_l1485_148585

theorem alex_singing_probability (p_sat : ℝ) 
  (h1 : ℝ → Prop) (h2 : ℝ → Prop) (h3 : ℝ → Prop) : Prop :=
  (h1 p_sat → (1 - p_sat) * 0.7 = 0.5) →
  (h2 p_sat → p_sat * 0 + (1 - p_sat) * 0.7 = 0.5) →
  (h3 p_sat → p_sat = 2 / 7) →
  p_sat = 2 / 7

-- The proof is omitted
theorem alex_singing_probability_proof : 
  ∃ (p_sat : ℝ) (h1 h2 h3 : ℝ → Prop), 
  alex_singing_probability p_sat h1 h2 h3 := by
  sorry

end alex_singing_probability_alex_singing_probability_proof_l1485_148585


namespace lucas_class_size_l1485_148504

theorem lucas_class_size : ∃! x : ℕ, 
  70 < x ∧ x < 120 ∧ 
  x % 6 = 4 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧
  x = 148 := by
  sorry

end lucas_class_size_l1485_148504


namespace prob_different_colors_7_5_l1485_148526

/-- The probability of drawing two chips of different colors from a bag with replacement -/
def prob_different_colors (red_chips green_chips : ℕ) : ℚ :=
  let total_chips := red_chips + green_chips
  let prob_red := red_chips / total_chips
  let prob_green := green_chips / total_chips
  2 * (prob_red * prob_green)

/-- Theorem stating that the probability of drawing two chips of different colors
    from a bag with 7 red chips and 5 green chips, with replacement, is 35/72 -/
theorem prob_different_colors_7_5 :
  prob_different_colors 7 5 = 35 / 72 := by
  sorry

end prob_different_colors_7_5_l1485_148526


namespace f_2002_eq_zero_l1485_148576

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_2002_eq_zero
  (h1 : is_even f)
  (h2 : f 2 = 0)
  (h3 : is_odd g)
  (h4 : ∀ x, g x = f (x - 1)) :
  f 2002 = 0 := by
  sorry

end f_2002_eq_zero_l1485_148576


namespace triangle_side_length_l1485_148588

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  C = 2 * A ∧
  Real.cos A = 3/4 ∧
  a * c * Real.cos B = 27/2 →
  b = 5 := by sorry

end triangle_side_length_l1485_148588


namespace centers_form_rectangle_l1485_148538

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : Prop := ∃ (C C1 C2 C3 C4 : Circle),
  -- C has radius 2
  C.radius = 2 ∧
  -- C1 and C2 have radius 1
  C1.radius = 1 ∧ C2.radius = 1 ∧
  -- C1 and C2 touch at the center of C
  C1.center = C.center + (1, 0) ∧ C2.center = C.center + (-1, 0) ∧
  -- C3 is inside C and touches C, C1, and C2
  (∃ x : ℝ, C3.radius = x ∧
    dist C3.center C.center = 2 - x ∧
    dist C3.center C1.center = 1 + x ∧
    dist C3.center C2.center = 1 + x) ∧
  -- C4 is inside C and touches C, C1, and C3
  (∃ y : ℝ, C4.radius = y ∧
    dist C4.center C.center = 2 - y ∧
    dist C4.center C1.center = 1 + y ∧
    dist C4.center C3.center = C3.radius + y)

-- Define what it means for four points to form a rectangle
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := dist p1 p2
  let d23 := dist p2 p3
  let d34 := dist p3 p4
  let d41 := dist p4 p1
  let d13 := dist p1 p3
  let d24 := dist p2 p4
  d12 = d34 ∧ d23 = d41 ∧ d13 = d24

-- Theorem statement
theorem centers_form_rectangle :
  problem_setup →
  ∃ (C C1 C3 C4 : Circle),
    form_rectangle C.center C1.center C3.center C4.center :=
sorry

end centers_form_rectangle_l1485_148538


namespace parabola_focus_l1485_148594

/-- The parabola equation: y^2 + 4x = 0 -/
def parabola_eq (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that the focus of the parabola y^2 + 4x = 0 is at (-1, 0) -/
theorem parabola_focus :
  ∃ (f : Focus), (f.x = -1 ∧ f.y = 0) ∧
  ∀ (x y : ℝ), parabola_eq x y → 
    (y^2 = 4 * (f.x - x) ∧ f.y = 0) :=
sorry

end parabola_focus_l1485_148594


namespace square_difference_formula_l1485_148561

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/19) : x^2 - y^2 = 9/323 := by
  sorry

end square_difference_formula_l1485_148561


namespace opposite_direction_speed_l1485_148509

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the speed of the other and their final distance. -/
theorem opposite_direction_speed 
  (pooja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : pooja_speed = 3) 
  (h2 : time = 4) 
  (h3 : final_distance = 32) : 
  ∃ (roja_speed : ℝ), roja_speed = 5 ∧ final_distance = (roja_speed + pooja_speed) * time :=
by sorry

end opposite_direction_speed_l1485_148509


namespace least_addition_for_divisibility_l1485_148563

theorem least_addition_for_divisibility (n : ℕ) : 
  (1024 + n) % 25 = 0 ∧ ∀ m : ℕ, m < n → (1024 + m) % 25 ≠ 0 ↔ n = 1 :=
by sorry

end least_addition_for_divisibility_l1485_148563


namespace fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1485_148500

theorem fraction_zero_at_minus_one (x : ℝ) :
  (x + 1) / (9 * x^2 - 74 * x + 9) = 0 ↔ x = -1 :=
by
  sorry

theorem denominator_nonzero_at_minus_one :
  9 * (-1)^2 - 74 * (-1) + 9 ≠ 0 :=
by
  sorry

theorem largest_x_for_zero_fraction :
  ∀ y > -1, (y + 1) / (9 * y^2 - 74 * y + 9) ≠ 0 :=
by
  sorry

end fraction_zero_at_minus_one_denominator_nonzero_at_minus_one_largest_x_for_zero_fraction_l1485_148500


namespace m_range_l1485_148582

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧
    m * x₁^2 - x₁ + m - 4 = 0 ∧
    m * x₂^2 - x₂ + m - 4 = 0

theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → Real.sqrt 2 + 1 ≤ m ∧ m < 4 :=
sorry

end m_range_l1485_148582


namespace fourth_score_calculation_l1485_148542

def score1 : ℝ := 65
def score2 : ℝ := 67
def score3 : ℝ := 76
def average_score : ℝ := 76.6
def num_subjects : ℕ := 4

theorem fourth_score_calculation :
  ∃ (score4 : ℝ),
    (score1 + score2 + score3 + score4) / num_subjects = average_score ∧
    score4 = 98.4 := by
  sorry

end fourth_score_calculation_l1485_148542


namespace kevin_has_eight_toads_l1485_148544

/-- The number of worms each toad eats daily -/
def worms_per_toad : ℕ := 3

/-- The time in minutes it takes Kevin to find one worm -/
def minutes_per_worm : ℕ := 15

/-- The total time in hours Kevin spends finding worms -/
def total_hours : ℕ := 6

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

/-- Calculates the number of toads Kevin has -/
def number_of_toads : ℕ :=
  (hours_to_minutes total_hours) / minutes_per_worm / worms_per_toad

theorem kevin_has_eight_toads : number_of_toads = 8 := by
  sorry

end kevin_has_eight_toads_l1485_148544


namespace average_score_theorem_l1485_148541

/-- The average score of a class given the proportions of students scoring different points -/
theorem average_score_theorem (p3 p2 p1 p0 : ℝ) 
  (h_p3 : p3 = 0.3) 
  (h_p2 : p2 = 0.5) 
  (h_p1 : p1 = 0.1) 
  (h_p0 : p0 = 0.1)
  (h_sum : p3 + p2 + p1 + p0 = 1) : 
  3 * p3 + 2 * p2 + 1 * p1 + 0 * p0 = 2 := by
  sorry

#check average_score_theorem

end average_score_theorem_l1485_148541


namespace apple_distribution_l1485_148513

theorem apple_distribution (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) : 
  total_apples = 9 → num_friends = 3 → total_apples / num_friends = apples_per_friend → apples_per_friend = 3 := by
  sorry

end apple_distribution_l1485_148513


namespace universal_set_determination_l1485_148558

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def complementA : Set Nat := {2, 4, 6}

theorem universal_set_determination :
  (A ⊆ U) ∧ (complementA ⊆ U) ∧ (A ∪ complementA = U) ∧ (A ∩ complementA = ∅) →
  U = {1, 2, 3, 4, 5, 6} :=
by sorry

end universal_set_determination_l1485_148558


namespace tan_2x_value_l1485_148552

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_2x_value (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + Real.cos x)
  (hf' : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 : ℝ) = -4/3 := by
sorry

end tan_2x_value_l1485_148552


namespace roots_and_inequality_solution_set_l1485_148589

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem roots_and_inequality_solution_set 
  (a b : ℝ) 
  (h1 : f a b (-1) = 0) 
  (h2 : f a b 2 = 0) :
  {x : ℝ | a * f a b (-2*x) > 0} = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by
sorry

end roots_and_inequality_solution_set_l1485_148589


namespace de_plus_ef_sum_l1485_148562

/-- Represents a polygon ABCDEF with specific properties -/
structure Polygon where
  area : ℝ
  ab : ℝ
  bc : ℝ
  fa : ℝ
  de_parallel_ab : Prop
  df_horizontal : ℝ

/-- Theorem stating the sum of DE and EF in the given polygon -/
theorem de_plus_ef_sum (p : Polygon) 
  (h1 : p.area = 75)
  (h2 : p.ab = 7)
  (h3 : p.bc = 10)
  (h4 : p.fa = 6)
  (h5 : p.de_parallel_ab)
  (h6 : p.df_horizontal = 8) :
  ∃ (de ef : ℝ), de + ef = 8.25 := by
  sorry

end de_plus_ef_sum_l1485_148562


namespace john_earnings_160_l1485_148533

/-- Calculates John's weekly streaming earnings --/
def johnWeeklyEarnings (daysOff : ℕ) (hoursPerDay : ℕ) (ratePerHour : ℕ) : ℕ :=
  let daysStreaming := 7 - daysOff
  let hoursPerWeek := daysStreaming * hoursPerDay
  hoursPerWeek * ratePerHour

/-- Theorem: John's weekly earnings are $160 --/
theorem john_earnings_160 :
  johnWeeklyEarnings 3 4 10 = 160 := by
  sorry

#eval johnWeeklyEarnings 3 4 10

end john_earnings_160_l1485_148533


namespace clare_remaining_money_l1485_148554

/-- Given Clare's initial money and her purchases, calculate the remaining money. -/
def remaining_money (initial_money bread_price milk_price bread_quantity milk_quantity : ℕ) : ℕ :=
  initial_money - (bread_price * bread_quantity + milk_price * milk_quantity)

/-- Theorem: Clare has $35 left after her purchases. -/
theorem clare_remaining_money :
  remaining_money 47 2 2 4 2 = 35 := by
  sorry

end clare_remaining_money_l1485_148554


namespace sum_of_last_two_digits_of_7_pow_1024_l1485_148581

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1024 is 17. -/
theorem sum_of_last_two_digits_of_7_pow_1024 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (7^1024 : ℕ) % 100 = 10 * a + b ∧ a + b = 17 := by
sorry

end sum_of_last_two_digits_of_7_pow_1024_l1485_148581


namespace second_article_loss_percentage_l1485_148574

/-- Proves that the loss percentage on the second article is 10% given the specified conditions --/
theorem second_article_loss_percentage
  (cost_price : ℝ)
  (profit_percent_first : ℝ)
  (net_profit_loss_percent : ℝ)
  (h1 : cost_price = 1000)
  (h2 : profit_percent_first = 10)
  (h3 : net_profit_loss_percent = 99.99999999999946) :
  let selling_price_first := cost_price * (1 + profit_percent_first / 100)
  let total_selling_price := 2 * cost_price * (1 + net_profit_loss_percent / 100)
  let selling_price_second := total_selling_price - selling_price_first
  let loss_second := cost_price - selling_price_second
  loss_second / cost_price * 100 = 10 := by
sorry


end second_article_loss_percentage_l1485_148574


namespace sequence_general_term_l1485_148516

/-- Given sequences {a_n} and {b_n} with initial conditions and recurrence relations,
    prove the general term formula for {b_n}. -/
theorem sequence_general_term
  (p q r : ℝ)
  (h_q_pos : q > 0)
  (h_p_gt_r : p > r)
  (h_r_pos : r > 0)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a_init : a 1 = p)
  (h_b_init : b 1 = q)
  (h_a_rec : ∀ n : ℕ, n ≥ 2 → a n = p * a (n - 1))
  (h_b_rec : ∀ n : ℕ, n ≥ 2 → b n = q * a (n - 1) + r * b (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → b n = (q * (p^n - r^n)) / (p - r) :=
by sorry

end sequence_general_term_l1485_148516


namespace chris_babysitting_hours_l1485_148517

/-- The number of hours Chris worked babysitting -/
def hours_worked : ℕ := 9

/-- The cost of the video game in dollars -/
def video_game_cost : ℕ := 60

/-- The cost of the candy in dollars -/
def candy_cost : ℕ := 5

/-- Chris's hourly rate for babysitting in dollars -/
def hourly_rate : ℕ := 8

/-- The amount of money Chris had left over after purchases -/
def money_left : ℕ := 7

theorem chris_babysitting_hours :
  hours_worked * hourly_rate = video_game_cost + candy_cost + money_left :=
by sorry

end chris_babysitting_hours_l1485_148517


namespace crackers_duration_l1485_148550

theorem crackers_duration (crackers_per_sandwich : ℕ) (sandwiches_per_night : ℕ)
  (sleeves_per_box : ℕ) (crackers_per_sleeve : ℕ) (num_boxes : ℕ)
  (h1 : crackers_per_sandwich = 2)
  (h2 : sandwiches_per_night = 5)
  (h3 : sleeves_per_box = 4)
  (h4 : crackers_per_sleeve = 28)
  (h5 : num_boxes = 5) :
  (num_boxes * sleeves_per_box * crackers_per_sleeve) / (crackers_per_sandwich * sandwiches_per_night) = 56 := by
  sorry

end crackers_duration_l1485_148550


namespace total_erasers_l1485_148531

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_erasers_l1485_148531


namespace problem_solution_l1485_148505

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  b / (a + b) + c / (b + c) + a / (c + a) = 6 := by
  sorry

end problem_solution_l1485_148505


namespace remainder_is_zero_l1485_148553

def divisors : List ℕ := [12, 15, 20, 54]
def least_number : ℕ := 540

theorem remainder_is_zero (n : ℕ) (h : n ∈ divisors) : 
  least_number % n = 0 := by sorry

end remainder_is_zero_l1485_148553


namespace special_polynomial_n_is_two_l1485_148577

/-- A polynomial of degree 2n satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) ∧
  (p (2 * n + 1) = -30)

/-- The theorem stating that n must be 2 for the given conditions -/
theorem special_polynomial_n_is_two :
  ∀ p : ℝ → ℝ, ∀ n : ℕ, SpecialPolynomial p n → n = 2 :=
by sorry

end special_polynomial_n_is_two_l1485_148577


namespace ceiling_sum_of_roots_l1485_148596

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_of_roots_l1485_148596


namespace triangle_angle_difference_l1485_148583

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real

-- Define the theorem
theorem triangle_angle_difference (t : Triangle) (h1 : t.Y = 2 * t.X) (h2 : t.X = 30) 
  (Z₁ Z₂ : Real) (h3 : Z₁ + Z₂ = t.Z) : Z₁ - Z₂ = 30 := by
  sorry


end triangle_angle_difference_l1485_148583


namespace cos_equality_for_specific_angles_l1485_148572

theorem cos_equality_for_specific_angles :
  ∀ n : ℤ, 0 ≤ n ∧ n ≤ 360 →
    (Real.cos (n * π / 180) = Real.cos (321 * π / 180) ↔ n = 39 ∨ n = 321) := by
  sorry

end cos_equality_for_specific_angles_l1485_148572


namespace ball_placement_theorem_l1485_148508

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process -/
def ballPlacement (step : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem (step : ℕ) :
  step = 1024 →
  ballPlacement step = sumDigits (toBase7 step) :=
sorry

end ball_placement_theorem_l1485_148508


namespace quadratic_function_properties_l1485_148540

/-- Quadratic function y = x^2 - 2tx + 3 -/
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  (f t 2 = 1 → t = 3/2) ∧
  (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 3 ∧ 
    (∀ x, x ∈ Set.Icc 0 3 → f t x ≥ f t x_min) ∧ 
    f t x_min = -2 → t = Real.sqrt 5) ∧
  (∀ (m a b : ℝ), 
    f t (m-2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 → 
    (3 < m ∧ m < 4) ∨ m > 6) := by sorry

end quadratic_function_properties_l1485_148540


namespace square_field_diagonal_l1485_148519

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 450 → diagonal = 30 → diagonal^2 = 2 * area :=
by
  sorry

end square_field_diagonal_l1485_148519


namespace cylinder_volume_in_sphere_l1485_148584

-- Define the sphere
def sphere_diameter : ℝ := 2

-- Define the cylinder
def cylinder_height : ℝ := 1

-- Theorem to prove
theorem cylinder_volume_in_sphere :
  let sphere_radius : ℝ := sphere_diameter / 2
  let base_radius : ℝ := sphere_radius
  let cylinder_volume : ℝ := Real.pi * base_radius^2 * (cylinder_height / 2)
  cylinder_volume = Real.pi / 2 := by
  sorry

end cylinder_volume_in_sphere_l1485_148584


namespace exists_coplanar_even_sum_l1485_148512

-- Define a cube as a set of 8 integers (representing the labels on vertices)
def Cube := Fin 8 → ℤ

-- Define a function to check if a set of four vertices is coplanar
def isCoplanar (v1 v2 v3 v4 : Fin 8) : Prop := sorry

-- Define a function to check if the sum of four integers is even
def sumIsEven (a b c d : ℤ) : Prop :=
  (a + b + c + d) % 2 = 0

-- Theorem statement
theorem exists_coplanar_even_sum (cube : Cube) :
  ∃ (v1 v2 v3 v4 : Fin 8), isCoplanar v1 v2 v3 v4 ∧ sumIsEven (cube v1) (cube v2) (cube v3) (cube v4) := by
  sorry

end exists_coplanar_even_sum_l1485_148512


namespace inequality_proof_l1485_148591

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (a + b) (min (b + c) (c + a)) > Real.sqrt 2)
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a / (b + c - a)^2 + b / (c + a - b)^2 + c / (a + b - c)^2 ≥ 3 / (a * b * c)^2 := by
  sorry

end inequality_proof_l1485_148591


namespace vasya_floor_l1485_148536

theorem vasya_floor (steps_per_floor : ℕ) (petya_steps : ℕ) (vasya_steps : ℕ) : 
  steps_per_floor * 2 = petya_steps → 
  vasya_steps = steps_per_floor * 4 → 
  5 = vasya_steps / steps_per_floor + 1 :=
by
  sorry

end vasya_floor_l1485_148536


namespace problem_solution_l1485_148573

def f (x : ℝ) : ℝ := |x - 1|

theorem problem_solution :
  (∃ (m : ℝ), m > 0 ∧
    (∀ x, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 →
    2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 ∧
    2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) := by
  sorry

end problem_solution_l1485_148573


namespace quadratic_function_and_triangle_area_l1485_148501

def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_function_and_triangle_area 
  (a b c : ℝ) 
  (h_opens_upward : a > 0)
  (h_not_origin : QuadraticFunction a b c 0 ≠ 0)
  (h_vertex : QuadraticFunction a b c 1 = -2)
  (x₁ x₂ : ℝ) 
  (h_roots : QuadraticFunction a b c x₁ = 0 ∧ QuadraticFunction a b c x₂ = 0)
  (h_y_intercept : (QuadraticFunction a b c 0)^2 = |x₁ * x₂|) :
  ((a = 1 ∧ b = -2 ∧ c = -1 ∧ (x₁ - x₂)^2 / 4 = 2) ∨
   (a = 1 + Real.sqrt 2 ∧ b = -(2 + 2 * Real.sqrt 2) ∧ c = Real.sqrt 2 - 1 ∧
    (x₁ - x₂)^2 / 4 = 2 * (Real.sqrt 2 - 1))) := by
  sorry

end quadratic_function_and_triangle_area_l1485_148501


namespace quadrilateral_bd_length_l1485_148597

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_bd_length (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 4 →
  length ABCD.B ABCD.C = 14 →
  length ABCD.C ABCD.D = 4 →
  length ABCD.D ABCD.A = 7 →
  ∃ (n : ℕ), length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 11 := by
sorry

end quadrilateral_bd_length_l1485_148597


namespace p_fourth_minus_one_divisible_by_ten_l1485_148525

theorem p_fourth_minus_one_divisible_by_ten (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  10 ∣ (p^4 - 1) := by
  sorry

end p_fourth_minus_one_divisible_by_ten_l1485_148525


namespace min_value_theorem_l1485_148559

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) ≥ 22 ∧
  ((x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) = 22 ↔ x = 3 ∧ y = 3) :=
by sorry

end min_value_theorem_l1485_148559


namespace set_operations_l1485_148522

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 3}

-- Define the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < 2 ∨ x ≥ 3}) := by
  sorry


end set_operations_l1485_148522


namespace annie_extracurricular_hours_l1485_148549

/-- The number of hours Annie spends on extracurriculars before midterms -/
def extracurricular_hours : ℕ :=
  let chess_hours : ℕ := 2
  let drama_hours : ℕ := 8
  let glee_hours : ℕ := 3
  let weekly_hours : ℕ := chess_hours + drama_hours + glee_hours
  let semester_weeks : ℕ := 12
  let midterm_weeks : ℕ := semester_weeks / 2
  let sick_weeks : ℕ := 2
  let active_weeks : ℕ := midterm_weeks - sick_weeks
  weekly_hours * active_weeks

/-- Theorem stating that Annie spends 52 hours on extracurriculars before midterms -/
theorem annie_extracurricular_hours : extracurricular_hours = 52 := by
  sorry

end annie_extracurricular_hours_l1485_148549


namespace max_sum_of_factors_l1485_148565

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ heart + club) → heart + club = 49 := by
  sorry

end max_sum_of_factors_l1485_148565


namespace integer_solutions_inequality_l1485_148551

theorem integer_solutions_inequality :
  ∀ x y z : ℤ,
  x^2 * y^2 + y^2 * z^2 + x^2 + z^2 - 38*(x*y + z) - 40*(y*z + x) + 4*x*y*z + 761 ≤ 0 →
  ((x = 6 ∧ y = 2 ∧ z = 7) ∨ (x = 20 ∧ y = 0 ∧ z = 19)) :=
by sorry

end integer_solutions_inequality_l1485_148551


namespace min_value_of_expression_l1485_148506

theorem min_value_of_expression (x y : ℝ) : (x * y - 2)^2 + (x + y - 1)^2 ≥ 2 := by
  sorry

end min_value_of_expression_l1485_148506


namespace coefficient_of_monomial_l1485_148560

def monomial : ℚ × (ℕ × ℕ × ℕ) := (-2/9, (1, 4, 2))

theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -2/9 := by sorry

end coefficient_of_monomial_l1485_148560


namespace solve_sock_problem_l1485_148545

def sock_problem (lisa_initial : ℕ) (sandra : ℕ) (total : ℕ) : Prop :=
  let cousin := sandra / 5
  let before_mom := lisa_initial + sandra + cousin
  ∃ (mom : ℕ), before_mom + mom = total

theorem solve_sock_problem :
  sock_problem 12 20 80 → ∃ (mom : ℕ), mom = 44 := by
  sorry

end solve_sock_problem_l1485_148545


namespace parabola_shift_theorem_l1485_148580

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * h, c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let shifted_left := shift_horizontal original 2
  let final := shift_vertical shifted_left (-3)
  final = Parabola.mk 1 (-4) (-2)  -- y = (x + 2)^2 - 2
  := by sorry

end parabola_shift_theorem_l1485_148580


namespace solution_to_equation_l1485_148521

theorem solution_to_equation (x : ℝ) :
  Real.sqrt (4 * x^2 + 4 * x + 1) - Real.sqrt (4 * x^2 - 12 * x + 9) = 4 →
  x ≥ 3/2 :=
by sorry

end solution_to_equation_l1485_148521


namespace orthogonal_vectors_x_value_l1485_148578

theorem orthogonal_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∀ i, i < 2 → a i * b i = 0) → x = 1/2 := by
  sorry

end orthogonal_vectors_x_value_l1485_148578
