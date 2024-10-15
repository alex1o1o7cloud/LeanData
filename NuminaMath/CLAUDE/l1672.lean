import Mathlib

namespace NUMINAMATH_CALUDE_roses_cut_correct_l1672_167271

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is correct -/
theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16

end NUMINAMATH_CALUDE_roses_cut_correct_l1672_167271


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l1672_167210

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the ratio of canoes to kayaks rented -/
theorem canoe_kayak_ratio (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 14)
  (h2 : rb.kayak_price = 15)
  (h3 : rb.total_revenue = 288)
  (h4 : rb.canoe_kayak_difference = 4)
  (h5 : ∃ (k : ℕ), rb.canoe_price * (k + rb.canoe_kayak_difference) + rb.kayak_price * k = rb.total_revenue) :
  ∃ (c k : ℕ), c = k + rb.canoe_kayak_difference ∧ c * rb.canoe_price + k * rb.kayak_price = rb.total_revenue ∧ c * 2 = k * 3 :=
by sorry

end NUMINAMATH_CALUDE_canoe_kayak_ratio_l1672_167210


namespace NUMINAMATH_CALUDE_model_c_sample_size_l1672_167251

/-- Calculates the number of units to be sampled from a specific model in stratified sampling. -/
def stratified_sample_size (total_units : ℕ) (sample_size : ℕ) (model_units : ℕ) : ℕ :=
  (model_units * sample_size) / total_units

/-- Theorem stating that the stratified sample size for Model C is 10 units. -/
theorem model_c_sample_size :
  let total_units : ℕ := 1400 + 5600 + 2000
  let sample_size : ℕ := 45
  let model_c_units : ℕ := 2000
  stratified_sample_size total_units sample_size model_c_units = 10 := by
  sorry

end NUMINAMATH_CALUDE_model_c_sample_size_l1672_167251


namespace NUMINAMATH_CALUDE_pipe_b_shut_time_l1672_167202

-- Define the rates at which pipes fill the tank
def pipe_a_rate : ℚ := 1
def pipe_b_rate : ℚ := 1 / 15

-- Define the time it takes for the tank to overflow
def overflow_time : ℚ := 1 / 2  -- 30 minutes = 0.5 hours

-- Define the theorem
theorem pipe_b_shut_time :
  let combined_rate := pipe_a_rate + pipe_b_rate
  let volume_filled_together := combined_rate * overflow_time
  let pipe_b_shut_time := 1 - volume_filled_together
  pipe_b_shut_time * 60 = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_b_shut_time_l1672_167202


namespace NUMINAMATH_CALUDE_novel_reading_difference_novel_reading_difference_proof_l1672_167285

theorem novel_reading_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jordan alexandre camille maxime =>
    jordan = 130 ∧
    alexandre = jordan / 10 ∧
    camille = 2 * alexandre ∧
    maxime = (jordan + alexandre + camille) / 2 - 5 →
    jordan - maxime = 51

-- Proof
theorem novel_reading_difference_proof :
  ∃ (jordan alexandre camille maxime : ℕ),
    novel_reading_difference jordan alexandre camille maxime :=
by
  sorry

end NUMINAMATH_CALUDE_novel_reading_difference_novel_reading_difference_proof_l1672_167285


namespace NUMINAMATH_CALUDE_fraction_square_value_l1672_167236

theorem fraction_square_value (x y : ℚ) (hx : x = 3) (hy : y = 5) :
  ((1 / y) / (1 / x))^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_value_l1672_167236


namespace NUMINAMATH_CALUDE_bike_ride_distance_l1672_167295

/-- Calculates the total distance traveled given a constant speed and time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem bike_ride_distance :
  let rate := 1.5 / 10  -- miles per minute
  let time := 40        -- minutes
  total_distance rate time = 6 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l1672_167295


namespace NUMINAMATH_CALUDE_prairie_total_area_l1672_167220

/-- The total area of a prairie given the dusted and untouched areas -/
theorem prairie_total_area (dusted_area untouched_area : ℕ) 
  (h1 : dusted_area = 64535)
  (h2 : untouched_area = 522) :
  dusted_area + untouched_area = 65057 := by
  sorry

#check prairie_total_area

end NUMINAMATH_CALUDE_prairie_total_area_l1672_167220


namespace NUMINAMATH_CALUDE_batch_size_l1672_167286

/-- The number of parts A can complete in one day -/
def a_rate : ℚ := 1 / 10

/-- The number of parts B can complete in one day -/
def b_rate : ℚ := 1 / 15

/-- The number of additional parts A completes compared to B in one day -/
def additional_parts : ℕ := 50

/-- The total number of parts in the batch -/
def total_parts : ℕ := 1500

theorem batch_size :
  (a_rate - b_rate) * total_parts = additional_parts := by sorry

end NUMINAMATH_CALUDE_batch_size_l1672_167286


namespace NUMINAMATH_CALUDE_frustum_volume_l1672_167245

/-- The volume of a frustum with specific conditions --/
theorem frustum_volume (r₁ r₂ : ℝ) (h : ℝ) : 
  r₁ = Real.sqrt 3 →
  r₂ = 3 * Real.sqrt 3 →
  h = 6 →
  (1/3 : ℝ) * (π * r₁^2 + π * r₂^2 + Real.sqrt (π^2 * r₁^2 * r₂^2)) * h = 78 * π := by
  sorry

#check frustum_volume

end NUMINAMATH_CALUDE_frustum_volume_l1672_167245


namespace NUMINAMATH_CALUDE_special_right_triangle_median_property_l1672_167215

/-- A right triangle with a special median property -/
structure SpecialRightTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- B is the right angle
  right_angle : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  -- BM is the median from B to AC
  M : ℝ × ℝ
  is_median : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- The special property BM² = AB·BC
  special_property : 
    ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 
    (((A.1 - B.1)^2 + (A.2 - B.2)^2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2))^(1/2)

/-- Theorem: In a SpecialRightTriangle, BM = 1/2 AC -/
theorem special_right_triangle_median_property (t : SpecialRightTriangle) :
  ((t.M.1 - t.B.1)^2 + (t.M.2 - t.B.2)^2) = 
  (1/4) * ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_median_property_l1672_167215


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1672_167266

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 200) (Finset.range 15)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1672_167266


namespace NUMINAMATH_CALUDE_meter_to_jumps_l1672_167268

-- Define the conversion factors
variable (a p q r s t u v : ℚ)

-- Define the relationships between units
axiom hops_to_skips : a * 1 = p
axiom jumps_to_hops : q * 1 = r
axiom skips_to_leaps : s * 1 = t
axiom leaps_to_meters : u * 1 = v

-- The theorem to prove
theorem meter_to_jumps : 1 = (u * s * a * q) / (p * v * t * r) :=
sorry

end NUMINAMATH_CALUDE_meter_to_jumps_l1672_167268


namespace NUMINAMATH_CALUDE_asphalt_work_hours_l1672_167265

/-- The number of hours per day the first group worked -/
def hours_per_day : ℝ := 8

/-- The number of men in the first group -/
def men_group1 : ℕ := 30

/-- The number of days the first group worked -/
def days_group1 : ℕ := 12

/-- The length of road asphalted by the first group in km -/
def road_length_group1 : ℝ := 1

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of hours per day the second group worked -/
def hours_per_day_group2 : ℕ := 15

/-- The number of days the second group worked -/
def days_group2 : ℝ := 19.2

/-- The length of road asphalted by the second group in km -/
def road_length_group2 : ℝ := 2

theorem asphalt_work_hours :
  hours_per_day * men_group1 * days_group1 * road_length_group2 =
  hours_per_day_group2 * men_group2 * days_group2 * road_length_group1 :=
by sorry

end NUMINAMATH_CALUDE_asphalt_work_hours_l1672_167265


namespace NUMINAMATH_CALUDE_complement_of_A_l1672_167230

-- Define the universal set U
def U : Set ℝ := {x | x < 4}

-- Define set A
def A : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_of_A : 
  (U \ A) = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1672_167230


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1672_167260

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  A + B + C = π →  -- Angle sum in a triangle
  (2*c - a) * Real.cos B = b * Real.cos A →  -- Given equation
  b = 6 →  -- Given condition
  c = 2*a →  -- Given condition
  B = π/3 ∧ (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1672_167260


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1672_167256

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 8000 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1672_167256


namespace NUMINAMATH_CALUDE_rectangle_area_l1672_167270

/-- Given a rectangle with perimeter 14 cm and diagonal 5 cm, its area is 12 square centimeters. -/
theorem rectangle_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 14) 
  (h_diagonal : l^2 + w^2 = 5^2) : l * w = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1672_167270


namespace NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l1672_167254

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add necessary fields here

-- Define an octahedron formed by midpoints of tetrahedron edges
structure MidpointOctahedron (t : RegularTetrahedron) where
  -- Add necessary fields here

-- Define volume calculation functions
def volume_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

def volume_octahedron (o : MidpointOctahedron t) : ℝ := sorry

-- Theorem statement
theorem midpoint_octahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (o : MidpointOctahedron t) : 
  volume_octahedron o / volume_tetrahedron t = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l1672_167254


namespace NUMINAMATH_CALUDE_smallest_y_abs_eq_l1672_167297

theorem smallest_y_abs_eq (y : ℝ) : (|2 * y + 6| = 18) → (∃ (z : ℝ), |2 * z + 6| = 18 ∧ z ≤ y) → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_abs_eq_l1672_167297


namespace NUMINAMATH_CALUDE_complex_equation_roots_l1672_167290

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Real.sqrt 7 * I) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Real.sqrt 7 * I) / 2
  (z₁^2 - z₁ = 3 - 7*I) ∧ (z₂^2 - z₂ = 3 - 7*I) := by
  sorry


end NUMINAMATH_CALUDE_complex_equation_roots_l1672_167290


namespace NUMINAMATH_CALUDE_trajectory_equation_l1672_167225

/-- The trajectory of a point M(x,y) such that its distance to the line x = 4 
    is twice its distance to the point (1,0) -/
def trajectory (x y : ℝ) : Prop :=
  (x - 4)^2 = ((x - 1)^2 + y^2) / 4

/-- The equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  trajectory x y ↔ 3 * x^2 + 30 * x - y^2 - 63 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1672_167225


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l1672_167237

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the set M parameterized by k
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset_condition :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l1672_167237


namespace NUMINAMATH_CALUDE_ada_original_seat_l1672_167234

-- Define the type for seats
inductive Seat : Type
  | one : Seat
  | two : Seat
  | three : Seat
  | four : Seat
  | five : Seat

-- Define the type for friends
inductive Friend : Type
  | ada : Friend
  | bea : Friend
  | ceci : Friend
  | dee : Friend
  | edie : Friend

-- Define the seating arrangement as a function from Friend to Seat
def SeatingArrangement : Type := Friend → Seat

-- Define what it means for a seat to be an end seat
def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.five

-- Define the movement of friends
def moveRight (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, 1 => Seat.five
  | _, _ => s  -- Default case: no movement or invalid movement

def moveLeft (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.four, 1 => Seat.three
  | Seat.five, 1 => Seat.four
  | _, _ => s  -- Default case: no movement or invalid movement

-- Theorem statement
theorem ada_original_seat (initial final : SeatingArrangement) :
  (∀ f : Friend, f ≠ Friend.ada → initial f ≠ Seat.five) →  -- No one except possibly Ada starts in seat 5
  (initial Friend.bea = moveLeft (final Friend.bea) 2) →   -- Bea moved 2 seats right
  (initial Friend.ceci = moveRight (final Friend.ceci) 1) → -- Ceci moved 1 seat left
  (initial Friend.dee = final Friend.edie ∧ initial Friend.edie = final Friend.dee) → -- Dee and Edie switched
  (isEndSeat (final Friend.ada)) →  -- Ada ends up in an end seat
  (initial Friend.ada = Seat.two) :=  -- Prove Ada started in seat 2
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l1672_167234


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l1672_167242

theorem no_three_digit_perfect_square_sum :
  ∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬∃ m : ℕ, m^2 = 111 * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l1672_167242


namespace NUMINAMATH_CALUDE_triangle_side_length_l1672_167227

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 9 →
  b = 2 * Real.sqrt 3 →
  C = 150 * π / 180 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1672_167227


namespace NUMINAMATH_CALUDE_composite_solid_volume_l1672_167274

/-- The volume of a composite solid consisting of a rectangular prism and a cylinder -/
theorem composite_solid_volume :
  ∀ (prism_length prism_width prism_height cylinder_radius cylinder_height overlap_volume : ℝ),
  prism_length = 2 →
  prism_width = 2 →
  prism_height = 1 →
  cylinder_radius = 1 →
  cylinder_height = 3 →
  overlap_volume = π / 2 →
  prism_length * prism_width * prism_height + π * cylinder_radius^2 * cylinder_height - overlap_volume = 4 + 5 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_composite_solid_volume_l1672_167274


namespace NUMINAMATH_CALUDE_isosceles_triangle_inscribed_circle_and_orthocenter_l1672_167240

/-- An isosceles triangle with unit-length legs -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ := 1

/-- The radius of the inscribed circle of an isosceles triangle -/
noncomputable def inscribedRadius (t : IsoscelesTriangle) : ℝ := sorry

/-- The orthocenter of an isosceles triangle -/
noncomputable def orthocenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- A point on the semicircle drawn on the base of the triangle -/
noncomputable def semicirclePoint (t : IsoscelesTriangle) : ℝ × ℝ := sorry

theorem isosceles_triangle_inscribed_circle_and_orthocenter 
  (t : IsoscelesTriangle) : 
  (∃ (max_t : IsoscelesTriangle), 
    (∀ (other_t : IsoscelesTriangle), inscribedRadius max_t ≥ inscribedRadius other_t) ∧
    max_t.base = Real.sqrt 5 - 1 ∧
    semicirclePoint max_t = orthocenter max_t) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_inscribed_circle_and_orthocenter_l1672_167240


namespace NUMINAMATH_CALUDE_john_unique_performance_l1672_167272

/-- Represents the Australian Senior Mathematics Competition (ASMC) -/
structure ASMC where
  total_questions : ℕ
  score_formula : ℕ → ℕ → ℕ
  score_uniqueness : ℕ → Prop

/-- John's performance in the ASMC -/
structure JohnPerformance where
  asmc : ASMC
  correct : ℕ
  wrong : ℕ

/-- Theorem stating that John's performance is unique given his score -/
theorem john_unique_performance (asmc : ASMC) (h : asmc.total_questions = 25) 
    (h_formula : asmc.score_formula = fun c w => 25 + 5 * c - 2 * w)
    (h_uniqueness : asmc.score_uniqueness = fun s => 
      ∀ c₁ w₁ c₂ w₂, s = asmc.score_formula c₁ w₁ → s = asmc.score_formula c₂ w₂ → 
      c₁ + w₁ ≤ asmc.total_questions → c₂ + w₂ ≤ asmc.total_questions → c₁ = c₂ ∧ w₁ = w₂) :
  ∃! (jp : JohnPerformance), 
    jp.asmc = asmc ∧ 
    jp.correct + jp.wrong ≤ asmc.total_questions ∧
    asmc.score_formula jp.correct jp.wrong = 100 ∧
    jp.correct = 19 ∧ jp.wrong = 10 := by
  sorry


end NUMINAMATH_CALUDE_john_unique_performance_l1672_167272


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1672_167217

/-- The eccentricity of a hyperbola with given properties is between 1 and 2√3/3 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * x = a * y ∨ b * x = -a * y}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ (p : ℝ × ℝ), p ∈ asymptotes ∩ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1672_167217


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l1672_167212

-- Define the fixed circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2

-- Define tangency condition
def isTangent (c₁ c₂ : ℝ → ℝ → Prop) (m : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y r, m x y r → (c₁ x y ∨ c₂ x y)

-- Main theorem
theorem trajectory_of_moving_circle :
  ∀ x y : ℝ, isTangent C₁ C₂ M → (x = 0 ∨ x^2 - y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l1672_167212


namespace NUMINAMATH_CALUDE_vector_equality_l1672_167204

/-- Given four non-overlapping points P, A, B, C on a plane, 
    if PA + PB + PC = 0 and AB + AC = m * AP, then m = 3 -/
theorem vector_equality (P A B C : ℝ × ℝ) (m : ℝ) 
  (h1 : (A.1 - P.1, A.2 - P.2) + (B.1 - P.1, B.2 - P.2) + (C.1 - P.1, C.2 - P.2) = (0, 0))
  (h2 : (B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2) = (m * (A.1 - P.1), m * (A.2 - P.2)))
  (h3 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l1672_167204


namespace NUMINAMATH_CALUDE_condition_a_geq_4_l1672_167213

theorem condition_a_geq_4 (a : ℝ) :
  (a ≥ 4 → ∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0) ∧
  ¬(∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0 → a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_condition_a_geq_4_l1672_167213


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1672_167207

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x / 3 < 1 - (x - 3) / 6 ∧ x < m) ↔ x < 3) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1672_167207


namespace NUMINAMATH_CALUDE_g_of_g_is_even_l1672_167222

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem g_of_g_is_even (g : ℝ → ℝ) (h : is_even_function g) : is_even_function (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_g_of_g_is_even_l1672_167222


namespace NUMINAMATH_CALUDE_main_theorem_l1672_167238

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

/-- The condition for y < 0 having no solution -/
def no_solution_condition (m : ℝ) : Prop :=
  ∀ x, y x m ≥ 0

/-- The solution set for y ≥ m when m > -2 -/
def solution_set (m : ℝ) : Set ℝ :=
  {x | y x m ≥ m}

theorem main_theorem :
  (∀ m : ℝ, no_solution_condition m ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (m = -1 → solution_set m = {x | x ≥ 1}) ∧
    (m > -1 → solution_set m = {x | x ≤ -1/(m+1) ∨ x ≥ 1}) ∧
    (-2 < m ∧ m < -1 → solution_set m = {x | 1 ≤ x ∧ x ≤ -1/(m+1)})) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l1672_167238


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l1672_167257

theorem two_numbers_with_given_means : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧
  b = (5 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l1672_167257


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1672_167206

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) :
  a 1 = 1 →
  a 4 = ∫ x in (1 : ℝ)..2, 3 * x^2 →
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l1672_167206


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l1672_167205

def num_balls : ℕ := 25
def num_bins : ℕ := 5

def count_distribution (d : List ℕ) : ℕ :=
  (List.prod (d.map (λ x => Nat.choose num_balls x))) / (Nat.factorial (List.length d))

theorem ball_distribution_ratio :
  let r := count_distribution [6, 7, 4, 4, 4] * Nat.factorial 5
  let s := count_distribution [5, 5, 5, 5, 5]
  (r : ℚ) / s = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l1672_167205


namespace NUMINAMATH_CALUDE_pen_pencil_difference_l1672_167296

theorem pen_pencil_difference (ratio_pens : ℕ) (ratio_pencils : ℕ) (total_pencils : ℕ) : 
  ratio_pens = 5 → ratio_pencils = 6 → total_pencils = 54 → 
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 9 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_difference_l1672_167296


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_range_l1672_167294

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The condition that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem stating that if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_range_l1672_167294


namespace NUMINAMATH_CALUDE_sample_size_calculation_l1672_167269

theorem sample_size_calculation (total_parts : ℕ) (prob_sampled : ℚ) (n : ℕ) : 
  total_parts = 200 → prob_sampled = 1/4 → n = (total_parts : ℚ) * prob_sampled → n = 50 := by
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l1672_167269


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1672_167288

/-- Given x = 11 * 36 * 42, prove that the smallest positive integer y 
    such that xy is a perfect cube is 5929 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 11 * 36 * 42) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ n : ℕ, x * y = n^3) ∧ 
    (∀ z : ℕ, z > 0 → z < y → ¬∃ m : ℕ, x * z = m^3) ∧
    y = 5929 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1672_167288


namespace NUMINAMATH_CALUDE_total_retail_price_proof_l1672_167249

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) : ℝ :=
  wholesale_price * (1 + profit_margin)

theorem total_retail_price_proof 
  (P Q R : ℝ)
  (discount1 discount2 discount3 : ℝ)
  (profit_margin1 profit_margin2 profit_margin3 : ℝ)
  (h1 : P = 90)
  (h2 : Q = 120)
  (h3 : R = 150)
  (h4 : discount1 = 0.10)
  (h5 : discount2 = 0.15)
  (h6 : discount3 = 0.20)
  (h7 : profit_margin1 = 0.20)
  (h8 : profit_margin2 = 0.25)
  (h9 : profit_margin3 = 0.30) :
  calculate_retail_price P profit_margin1 +
  calculate_retail_price Q profit_margin2 +
  calculate_retail_price R profit_margin3 = 453 := by
sorry

end NUMINAMATH_CALUDE_total_retail_price_proof_l1672_167249


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_green_is_one_third_l1672_167258

-- Define the number of pens of each color
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the total number of pens
def total_pens : ℕ := green_pens + black_pens + red_pens

-- Define the probability of picking a pen that is neither red nor green
def prob_neither_red_nor_green : ℚ := black_pens / total_pens

-- Theorem statement
theorem prob_neither_red_nor_green_is_one_third :
  prob_neither_red_nor_green = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_green_is_one_third_l1672_167258


namespace NUMINAMATH_CALUDE_scaled_triangle_not_valid_l1672_167292

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The original triangle PQR -/
def original_triangle : Triangle :=
  { a := 15, b := 20, c := 25 }

/-- The scaled triangle PQR -/
def scaled_triangle : Triangle :=
  { a := 3 * original_triangle.a,
    b := 2 * original_triangle.b,
    c := original_triangle.c }

/-- Theorem stating that the scaled triangle is not valid -/
theorem scaled_triangle_not_valid :
  ¬(is_valid_triangle scaled_triangle) :=
sorry

end NUMINAMATH_CALUDE_scaled_triangle_not_valid_l1672_167292


namespace NUMINAMATH_CALUDE_team_size_l1672_167221

theorem team_size (average_age : ℝ) (leader_age : ℝ) (average_age_without_leader : ℝ) 
  (h1 : average_age = 25)
  (h2 : leader_age = 45)
  (h3 : average_age_without_leader = 23) :
  ∃ n : ℕ, n * average_age = (n - 1) * average_age_without_leader + leader_age ∧ n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_team_size_l1672_167221


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1672_167218

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1 / 3)
  (h2 : water_ratio = 1 / 2)
  (h3 : gravel_weight = 8) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧ 
    total_weight = 48 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1672_167218


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l1672_167248

theorem trig_expression_equals_four : 
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l1672_167248


namespace NUMINAMATH_CALUDE_machine_subtraction_l1672_167291

theorem machine_subtraction (initial : ℕ) (added : ℕ) (subtracted : ℕ) (result : ℕ) :
  initial = 26 →
  added = 15 →
  result = 35 →
  initial + added - subtracted = result →
  subtracted = 6 := by
sorry

end NUMINAMATH_CALUDE_machine_subtraction_l1672_167291


namespace NUMINAMATH_CALUDE_min_value_expression_l1672_167298

theorem min_value_expression (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  ∃ (min_x : ℝ), min_x = 15 ∧
    ∀ y, a ≤ y ∧ y ≤ 15 →
      |y - a| + |y - 15| + |y - a - 15| ≥ |min_x - a| + |min_x - 15| + |min_x - a - 15| ∧
      |min_x - a| + |min_x - 15| + |min_x - a - 15| = 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1672_167298


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l1672_167224

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_ineq : |a*b - a*c| > |b^2 - a*c| + |a*b - c^2|)
  : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l1672_167224


namespace NUMINAMATH_CALUDE_set_operations_l1672_167209

def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

theorem set_operations :
  (A ∪ (B ∩ C) = {0, 1, 2, 3}) ∧
  (A ∩ (U \ (B ∪ C)) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1672_167209


namespace NUMINAMATH_CALUDE_root_sum_arctan_l1672_167203

theorem root_sum_arctan (x₁ x₂ : ℝ) (α β : ℝ) : 
  x₁^2 + 3 * Real.sqrt 3 * x₁ + 4 = 0 →
  x₂^2 + 3 * Real.sqrt 3 * x₂ + 4 = 0 →
  α = Real.arctan x₁ →
  β = Real.arctan x₂ →
  α + β = π / 3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_arctan_l1672_167203


namespace NUMINAMATH_CALUDE_work_completion_ratio_l1672_167287

/-- Given that A can finish a work in 18 days and that A and B working together
    can finish 1/6 of the work in a day, prove that the ratio of the time taken
    by B to finish the work alone to the time taken by A is 1/2. -/
theorem work_completion_ratio (a b : ℝ) (ha : a = 18) 
    (hab : 1 / a + 1 / b = 1 / 6) : b / a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_ratio_l1672_167287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1672_167259

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₃ + a₁₁ = 22, then a₇ = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 3 + a 11 = 22) : 
  a 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1672_167259


namespace NUMINAMATH_CALUDE_crayon_distribution_sum_l1672_167226

def arithmeticSequenceSum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem crayon_distribution_sum :
  arithmeticSequenceSum 18 12 2 = 522 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_sum_l1672_167226


namespace NUMINAMATH_CALUDE_range_of_a_l1672_167277

/-- The proposition p: The equation ax^2 + ax - 2 = 0 has a solution on the interval [-1, 1] -/
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a * x^2 + a * x - 2 = 0

/-- The proposition q: There is exactly one real number x such that x^2 + 2ax + 2a ≤ 0 -/
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

/-- The function f(x) = ax^2 + ax - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 2

theorem range_of_a (a : ℝ) :
  (¬(p a ∨ q a)) ∧ (f a (-1) = -2) ∧ (f a 0 = -2) →
  (-8 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1672_167277


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1672_167243

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (h1 : a ≠ 11)
  (h2 : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry


end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1672_167243


namespace NUMINAMATH_CALUDE_discount_difference_l1672_167263

theorem discount_difference : 
  let original_bill : ℝ := 10000
  let single_discount_rate : ℝ := 0.4
  let first_successive_discount_rate : ℝ := 0.36
  let second_successive_discount_rate : ℝ := 0.04
  let single_discounted_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discounted_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted_amount - single_discounted_amount = 144 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1672_167263


namespace NUMINAMATH_CALUDE_shaded_area_square_l1672_167235

theorem shaded_area_square (a : ℝ) (h : a = 4) : 
  let square_area := a ^ 2
  let shaded_area := square_area / 2
  shaded_area = 8 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_l1672_167235


namespace NUMINAMATH_CALUDE_sum_of_abs_coeff_equals_729_l1672_167211

/-- Given a polynomial p(x) = a₆x⁶ + a₅x⁵ + ... + a₁x + a₀ that equals (2x-1)⁶,
    the sum of the absolute values of its coefficients is 729. -/
theorem sum_of_abs_coeff_equals_729 (a : Fin 7 → ℤ) : 
  (∀ x, (2*x - 1)^6 = a 6 * x^6 + a 5 * x^5 + a 4 * x^4 + a 3 * x^3 + a 2 * x^2 + a 1 * x + a 0) →
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 729) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_coeff_equals_729_l1672_167211


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l1672_167278

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2 - a - 13) ↔ 
  (a ≥ -Real.sqrt 14 ∧ a ≤ 1 + Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l1672_167278


namespace NUMINAMATH_CALUDE_triangular_prism_tetrahedra_l1672_167247

/-- The number of vertices in a triangular prism -/
def triangular_prism_vertices : ℕ := 6

/-- The number of distinct tetrahedra that can be formed using the vertices of a triangular prism -/
def distinct_tetrahedra (n : ℕ) : ℕ := Nat.choose n 4 - 3

theorem triangular_prism_tetrahedra :
  distinct_tetrahedra triangular_prism_vertices = 12 := by sorry

end NUMINAMATH_CALUDE_triangular_prism_tetrahedra_l1672_167247


namespace NUMINAMATH_CALUDE_square_side_length_is_twenty_l1672_167208

/-- The side length of a square that can contain specific numbers of square tiles of different sizes -/
def square_side_length : ℕ := 
  let one_by_one := 4
  let two_by_two := 8
  let three_by_three := 12
  let four_by_four := 16
  let total_area := one_by_one * 1^2 + two_by_two * 2^2 + three_by_three * 3^2 + four_by_four * 4^2
  Nat.sqrt total_area

/-- Theorem stating that the side length of the square is 20 -/
theorem square_side_length_is_twenty : square_side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_is_twenty_l1672_167208


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1672_167299

def A : Set ℕ := {2, 3}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1672_167299


namespace NUMINAMATH_CALUDE_girls_boys_difference_l1672_167241

theorem girls_boys_difference (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end NUMINAMATH_CALUDE_girls_boys_difference_l1672_167241


namespace NUMINAMATH_CALUDE_ellipse_properties_l1672_167264

/-- An ellipse with specific properties -/
structure SpecificEllipse where
  foci_on_y_axis : Bool
  center_at_origin : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ

/-- A point satisfying a specific condition -/
structure SpecialPoint where
  coords : ℝ × ℝ
  condition : Bool

/-- Theorem about the specific ellipse and related geometric properties -/
theorem ellipse_properties (e : SpecificEllipse) (l : Line) (m : SpecialPoint) :
  e.foci_on_y_axis ∧
  e.center_at_origin ∧
  e.minor_axis_length = 2 * Real.sqrt 3 ∧
  e.eccentricity = 1 / 2 ∧
  l.point = (0, 3) ∧
  m.coords = (2, 0) ∧
  m.condition →
  (∃ (x y : ℝ), y^2 / 4 + x^2 / 3 = 1) ∧
  (∃ (d : ℝ), 0 ≤ d ∧ d < (48 + 8 * Real.sqrt 15) / 21) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1672_167264


namespace NUMINAMATH_CALUDE_no_divisible_polynomial_values_l1672_167231

theorem no_divisible_polynomial_values : ¬∃ (m n : ℤ), 
  0 < m ∧ m < n ∧ 
  (n ∣ (m^2 + m - 70)) ∧ 
  ((n + 1) ∣ ((m + 1)^2 + (m + 1) - 70)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_polynomial_values_l1672_167231


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1672_167232

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1672_167232


namespace NUMINAMATH_CALUDE_orange_cost_18_pounds_l1672_167228

/-- Calculates the cost of oranges given a rate and a desired weight -/
def orangeCost (ratePrice : ℚ) (rateWeight : ℚ) (desiredWeight : ℚ) : ℚ :=
  (ratePrice / rateWeight) * desiredWeight

theorem orange_cost_18_pounds :
  orangeCost 5 6 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_18_pounds_l1672_167228


namespace NUMINAMATH_CALUDE_lost_ship_depth_l1672_167216

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 2400 feet below sea level. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 30  -- feet per minute
  let time_taken : ℝ := 80    -- minutes
  depth_of_lost_ship descent_rate time_taken = 2400 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l1672_167216


namespace NUMINAMATH_CALUDE_complex_number_properties_l1672_167280

/-- Given two complex numbers z₁ and z₂ with unit magnitude, prove specific values for z₁ and z₂
    when their difference is given, and prove the value of their product when their sum is given. -/
theorem complex_number_properties (z₁ z₂ : ℂ) :
  (Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1) →
  (z₁ - z₂ = Complex.mk (Real.sqrt 6 / 3) (Real.sqrt 3 / 3) →
    z₁ = Complex.mk ((Real.sqrt 6 + 3) / 6) ((Real.sqrt 3 - 3 * Real.sqrt 2) / 6) ∧
    z₂ = Complex.mk ((-Real.sqrt 6 + 3) / 6) ((-Real.sqrt 3 - 3 * Real.sqrt 2) / 6)) ∧
  (z₁ + z₂ = Complex.mk (12/13) (-5/13) →
    z₁ * z₂ = Complex.mk (119/169) (-120/169)) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1672_167280


namespace NUMINAMATH_CALUDE_initial_workers_l1672_167281

theorem initial_workers (total : ℕ) (increase_percent : ℚ) : 
  total = 1065 → increase_percent = 25 / 100 → 
  ∃ initial : ℕ, initial * (1 + increase_percent) = total ∧ initial = 852 :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_l1672_167281


namespace NUMINAMATH_CALUDE_discount_amount_l1672_167239

/-- Given a shirt with an original price and a discounted price, 
    the discount amount is the difference between the two prices. -/
theorem discount_amount (original_price discounted_price : ℕ) :
  original_price = 22 →
  discounted_price = 16 →
  original_price - discounted_price = 6 := by
sorry

end NUMINAMATH_CALUDE_discount_amount_l1672_167239


namespace NUMINAMATH_CALUDE_concert_ticket_price_l1672_167275

theorem concert_ticket_price :
  ∀ (ticket_price : ℚ),
    (2 * ticket_price) +                    -- Cost of two tickets
    (0.15 * 2 * ticket_price) +             -- 15% processing fee
    10 +                                    -- Parking fee
    (2 * 5) =                               -- Entrance fee for two people
    135 →                                   -- Total cost
    ticket_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l1672_167275


namespace NUMINAMATH_CALUDE_stanleys_distance_difference_l1672_167246

theorem stanleys_distance_difference (run_distance walk_distance : ℝ) : 
  run_distance = 0.4 → walk_distance = 0.2 → run_distance - walk_distance = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_stanleys_distance_difference_l1672_167246


namespace NUMINAMATH_CALUDE_teams_bc_work_time_l1672_167223

-- Define the workload of projects
def project_a_workload : ℝ := 1
def project_b_workload : ℝ := 1.25

-- Define the time it takes for each team to complete Project A
def team_a_time : ℝ := 20
def team_b_time : ℝ := 24
def team_c_time : ℝ := 30

-- Define variables for the unknown times
def time_bc_together : ℝ := 15
def time_c_with_a : ℝ := 20 -- This is not given, but we need it for the theorem

theorem teams_bc_work_time :
  (time_bc_together / team_b_time + time_bc_together / team_c_time + time_c_with_a / team_b_time = project_b_workload) ∧
  (time_bc_together / team_a_time + time_c_with_a / team_c_time + time_c_with_a / team_a_time = project_a_workload) :=
by sorry

end NUMINAMATH_CALUDE_teams_bc_work_time_l1672_167223


namespace NUMINAMATH_CALUDE_hcl_equals_h2o_l1672_167252

-- Define the chemical reaction
structure ChemicalReaction where
  hcl : ℝ  -- moles of Hydrochloric acid
  nahco3 : ℝ  -- moles of Sodium bicarbonate
  h2o : ℝ  -- moles of Water formed

-- Define the conditions of the problem
def reaction_conditions (r : ChemicalReaction) : Prop :=
  r.nahco3 = 1 ∧ r.h2o = 1

-- Theorem statement
theorem hcl_equals_h2o (r : ChemicalReaction) 
  (h : reaction_conditions r) : r.hcl = r.h2o := by
  sorry

#check hcl_equals_h2o

end NUMINAMATH_CALUDE_hcl_equals_h2o_l1672_167252


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1672_167253

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1672_167253


namespace NUMINAMATH_CALUDE_problem_solution_l1672_167276

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 6) : 
  q = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1672_167276


namespace NUMINAMATH_CALUDE_division_addition_problem_l1672_167282

theorem division_addition_problem : (-144) / (-36) + 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_problem_l1672_167282


namespace NUMINAMATH_CALUDE_function_divisibility_property_l1672_167233

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_divisibility_property 
  (f : ℤ → ℕ) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), is_divisible (f m) (f n) → is_divisible (f m) (f n) :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l1672_167233


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1672_167244

theorem algebraic_expression_value (x : ℝ) : 
  x^2 + 2*x + 7 = 6 → 4*x^2 + 8*x - 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1672_167244


namespace NUMINAMATH_CALUDE_binomial_zero_binomial_312_0_l1672_167214

theorem binomial_zero (n : ℕ) : Nat.choose n 0 = 1 := by sorry

theorem binomial_312_0 : Nat.choose 312 0 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_zero_binomial_312_0_l1672_167214


namespace NUMINAMATH_CALUDE_study_time_difference_l1672_167283

-- Define the study times
def kwame_hours : ℝ := 2.5
def connor_hours : ℝ := 1.5
def lexia_minutes : ℝ := 97

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem to prove
theorem study_time_difference : 
  (kwame_hours * minutes_per_hour + connor_hours * minutes_per_hour) - lexia_minutes = 143 := by
  sorry

end NUMINAMATH_CALUDE_study_time_difference_l1672_167283


namespace NUMINAMATH_CALUDE_sphere_radius_from_cross_sections_l1672_167219

theorem sphere_radius_from_cross_sections (r : ℝ) (h₁ h₂ : ℝ) : 
  h₁ > h₂ →
  h₁ - h₂ = 1 →
  π * (r^2 - h₁^2) = 5 * π →
  π * (r^2 - h₂^2) = 8 * π →
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_cross_sections_l1672_167219


namespace NUMINAMATH_CALUDE_percentage_with_both_colors_l1672_167250

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  totalFlags : ℕ
  bluePercentage : ℚ
  redPercentage : ℚ
  bothPercentage : ℚ

/-- Theorem stating the percentage of children with both color flags -/
theorem percentage_with_both_colors (fd : FlagDistribution) :
  fd.totalFlags % 2 = 0 ∧
  fd.bluePercentage = 60 / 100 ∧
  fd.redPercentage = 45 / 100 ∧
  fd.bluePercentage + fd.redPercentage > 1 →
  fd.bothPercentage = 5 / 100 := by
  sorry

#check percentage_with_both_colors

end NUMINAMATH_CALUDE_percentage_with_both_colors_l1672_167250


namespace NUMINAMATH_CALUDE_b_is_killer_l1672_167261

-- Define the characters
inductive Character : Type
| A : Character
| B : Character
| C : Character

-- Define the actions
def poisoned_water (x y : Character) : Prop := x = Character.A ∧ y = Character.C
def made_hole (x y : Character) : Prop := x = Character.B ∧ y = Character.C
def died_of_thirst (x : Character) : Prop := x = Character.C

-- Define the killer
def is_killer (x : Character) : Prop := x = Character.B

-- Theorem statement
theorem b_is_killer 
  (h1 : poisoned_water Character.A Character.C)
  (h2 : made_hole Character.B Character.C)
  (h3 : died_of_thirst Character.C) :
  is_killer Character.B :=
sorry

end NUMINAMATH_CALUDE_b_is_killer_l1672_167261


namespace NUMINAMATH_CALUDE_fourth_sample_seat_number_l1672_167289

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : Finset ℕ
  interval : ℕ

/-- The theorem to prove -/
theorem fourth_sample_seat_number
  (s : SystematicSampling)
  (h_total : s.total_students = 56)
  (h_size : s.sample_size = 4)
  (h_known : s.known_samples = {3, 17, 45})
  (h_interval : s.interval = s.total_students / s.sample_size) :
  ∃ (n : ℕ), n ∈ s.known_samples ∧ (n + s.interval) % s.total_students = 31 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_seat_number_l1672_167289


namespace NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_l1672_167267

theorem cos_minus_sin_seventeen_fourths_pi : 
  Real.cos (-17 * Real.pi / 4) - Real.sin (-17 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_l1672_167267


namespace NUMINAMATH_CALUDE_power_equation_solution_l1672_167279

theorem power_equation_solution : ∃ x : ℕ, 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^x ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1672_167279


namespace NUMINAMATH_CALUDE_product_zero_from_sum_conditions_l1672_167201

theorem product_zero_from_sum_conditions (x y z w : ℝ) 
  (sum_condition : x + y + z + w = 0)
  (power_sum_condition : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_from_sum_conditions_l1672_167201


namespace NUMINAMATH_CALUDE_nat_less_than_5_finite_int_solution_set_nonempty_l1672_167284

-- Define the set of natural numbers less than 5
def nat_less_than_5 : Set ℕ := {n | n < 5}

-- Define the set of integers satisfying 2x + 1 > 7
def int_solution_set : Set ℤ := {x | 2 * x + 1 > 7}

-- Theorem 1: The set of natural numbers less than 5 is finite
theorem nat_less_than_5_finite : Finite nat_less_than_5 := by sorry

-- Theorem 2: The set of integers satisfying 2x + 1 > 7 is non-empty
theorem int_solution_set_nonempty : Set.Nonempty int_solution_set := by sorry

end NUMINAMATH_CALUDE_nat_less_than_5_finite_int_solution_set_nonempty_l1672_167284


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l1672_167293

theorem circle_circumference_increase (d : Real) : 
  let increase_in_diameter : Real := 2 * Real.pi
  let original_circumference : Real := Real.pi * d
  let new_circumference : Real := Real.pi * (d + increase_in_diameter)
  let Q : Real := new_circumference - original_circumference
  Q = 2 * Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l1672_167293


namespace NUMINAMATH_CALUDE_parabola_expression_l1672_167262

/-- A parabola that intersects the x-axis at (-1,0) and (2,0) and has the same shape and direction of opening as y = -2x² -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * (-1)^2 + b * (-1) + c = 0
  root2 : a * 2^2 + b * 2 + c = 0
  shape : a = -2

/-- The expression of the parabola is y = -2x² + 2x + 4 -/
theorem parabola_expression (p : Parabola) : p.a = -2 ∧ p.b = 2 ∧ p.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_expression_l1672_167262


namespace NUMINAMATH_CALUDE_polynomial_equality_l1672_167200

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1672_167200


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l1672_167255

/-- Proves the number of cookie packs Tory sold to his neighbor -/
theorem tory_cookie_sales (total : ℕ) (grandmother : ℕ) (uncle : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 50)
  (h2 : grandmother = 12)
  (h3 : uncle = 7)
  (h4 : left_to_sell = 26) :
  total - left_to_sell - (grandmother + uncle) = 5 := by
  sorry

#check tory_cookie_sales

end NUMINAMATH_CALUDE_tory_cookie_sales_l1672_167255


namespace NUMINAMATH_CALUDE_hulk_jump_distance_l1672_167273

def jump_distance (n : ℕ) : ℝ := 3 * (2 ^ (n - 1))

theorem hulk_jump_distance :
  (∀ k < 11, jump_distance k ≤ 3000) ∧ jump_distance 11 > 3000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_distance_l1672_167273


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l1672_167229

/-- Proves that the ratio of monthly savings to total monthly earnings is 1/2 --/
theorem savings_to_earnings_ratio 
  (car_washing_earnings : ℕ) 
  (dog_walking_earnings : ℕ) 
  (months_to_save : ℕ) 
  (total_savings : ℕ) 
  (h1 : car_washing_earnings = 20)
  (h2 : dog_walking_earnings = 40)
  (h3 : months_to_save = 5)
  (h4 : total_savings = 150) :
  (total_savings / months_to_save) / (car_washing_earnings + dog_walking_earnings) = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l1672_167229
