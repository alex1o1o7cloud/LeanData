import Mathlib

namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l947_94769

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l947_94769


namespace NUMINAMATH_CALUDE_jeremy_cannot_be_sure_l947_94724

theorem jeremy_cannot_be_sure (n : ℕ) : ∃ (remaining_permutations : ℝ), 
  remaining_permutations > 1 ∧ 
  remaining_permutations = (2^n).factorial / 2^(n * 2^(n-1)) := by
  sorry

#check jeremy_cannot_be_sure

end NUMINAMATH_CALUDE_jeremy_cannot_be_sure_l947_94724


namespace NUMINAMATH_CALUDE_remainder_problem_l947_94745

theorem remainder_problem : (((1234567 % 135) * 5) % 27) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l947_94745


namespace NUMINAMATH_CALUDE_batsman_average_l947_94728

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 16 * previous_average ∧
  (previous_total + 65 : ℚ) / 17 = previous_average + 3 →
  (previous_total + 65 : ℚ) / 17 = 17 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l947_94728


namespace NUMINAMATH_CALUDE_linear_equation_condition_l947_94798

theorem linear_equation_condition (m : ℝ) :
  (∃ x, (3*m - 1)*x + 9 = 0) ∧ (∀ x y, (3*m - 1)*x + 9 = 0 ∧ (3*m - 1)*y + 9 = 0 → x = y) →
  m ≠ 1/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l947_94798


namespace NUMINAMATH_CALUDE_binary_1010011_conversion_l947_94790

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : ℕ) : String :=
  let rec aux (m : ℕ) : List Char :=
    if m = 0 then []
    else
      let digit := m % 16
      let char := if digit < 10 then Char.ofNat (digit + 48) else Char.ofNat (digit + 55)
      char :: aux (m / 16)
  String.mk (aux n).reverse

/-- The binary number 1010011₂ -/
def binary_1010011 : List Bool := [true, true, false, false, true, false, true]

theorem binary_1010011_conversion :
  (binary_to_decimal binary_1010011 = 83) ∧
  (decimal_to_hex (binary_to_decimal binary_1010011) = "53") := by
  sorry

end NUMINAMATH_CALUDE_binary_1010011_conversion_l947_94790


namespace NUMINAMATH_CALUDE_power_sum_equality_l947_94765

theorem power_sum_equality : (-1)^53 + 2^(5^3 - 2^3 + 3^2) = 2^126 - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l947_94765


namespace NUMINAMATH_CALUDE_min_voters_for_giraffe_contest_l947_94719

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure :=
  (total_voters : ℕ)
  (num_districts : ℕ)
  (num_sections_per_district : ℕ)
  (voters_per_section : ℕ)
  (h_total : total_voters = num_districts * num_sections_per_district * voters_per_section)

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let sections_to_win := (vs.num_sections_per_district + 1) / 2
  let voters_to_win_section := (vs.voters_per_section + 1) / 2
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating the minimum number of voters required to win the contest -/
theorem min_voters_for_giraffe_contest :
  ∀ (vs : VotingStructure),
  vs.total_voters = 105 ∧
  vs.num_districts = 5 ∧
  vs.num_sections_per_district = 7 ∧
  vs.voters_per_section = 3 →
  min_voters_to_win vs = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  num_sections_per_district := 7,
  voters_per_section := 3,
  h_total := rfl
}

end NUMINAMATH_CALUDE_min_voters_for_giraffe_contest_l947_94719


namespace NUMINAMATH_CALUDE_calculator_transformation_l947_94710

/-- Transformation function for the calculator -/
def transform (a b : Int) : Int × Int :=
  match (a + b) % 4 with
  | 0 => (a + 1, b)
  | 1 => (a, b + 1)
  | 2 => (a - 1, b)
  | _ => (a, b - 1)

/-- Apply the transformation n times -/
def transformN (n : Nat) (a b : Int) : Int × Int :=
  match n with
  | 0 => (a, b)
  | n + 1 => 
    let (x, y) := transformN n a b
    transform x y

theorem calculator_transformation :
  transformN 6 1 12 = (-2, 15) :=
by sorry

end NUMINAMATH_CALUDE_calculator_transformation_l947_94710


namespace NUMINAMATH_CALUDE_ln2_greatest_l947_94789

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem ln2_greatest (h1 : ∀ x y : ℝ, x < y → ln x < ln y) (h2 : (2 : ℝ) < Real.exp 1) :
  ln 2 > (ln 2)^2 ∧ ln 2 > ln (ln 2) ∧ ln 2 > ln (Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ln2_greatest_l947_94789


namespace NUMINAMATH_CALUDE_magnitude_of_b_l947_94786

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 2√2 -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  let angle := 3 * π / 4
  a = (-3, 4) →
  a.fst * b.fst + a.snd * b.snd = -10 →
  Real.sqrt (b.fst ^ 2 + b.snd ^ 2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_b_l947_94786


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l947_94758

/-- Given an arithmetic sequence with first term 2 and common difference 5,
    the 50th term of this sequence is 247. -/
theorem arithmetic_sequence_50th_term :
  let a : ℕ → ℕ := λ n => 2 + (n - 1) * 5
  a 50 = 247 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l947_94758


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l947_94772

theorem contrapositive_equivalence (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l947_94772


namespace NUMINAMATH_CALUDE_coin_problem_l947_94755

theorem coin_problem (x : ℕ) : 
  (x + (x + 3) + (20 - 2*x) = 23) →  -- Total coins
  (5*x + 10*(x + 3) + 25*(20 - 2*x) = 320) →  -- Total value
  (20 - 2*x) - x = 2  -- Difference between 25-cent and 5-cent coins
  := by sorry

end NUMINAMATH_CALUDE_coin_problem_l947_94755


namespace NUMINAMATH_CALUDE_equation_solution_l947_94733

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l947_94733


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l947_94744

theorem parametric_to_ordinary_equation :
  ∀ (θ : ℝ) (x y : ℝ),
    x = Real.cos θ ^ 2 →
    y = 2 * Real.sin θ ^ 2 →
    2 * x + y - 2 = 0 ∧ x ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l947_94744


namespace NUMINAMATH_CALUDE_job_duration_l947_94730

theorem job_duration (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) (absent_days : ℕ) :
  daily_wage = 10 →
  daily_fine = 2 →
  total_earnings = 216 →
  absent_days = 7 →
  ∃ (work_days : ℕ), work_days * daily_wage - absent_days * daily_fine = total_earnings ∧ work_days = 23 :=
by sorry

end NUMINAMATH_CALUDE_job_duration_l947_94730


namespace NUMINAMATH_CALUDE_abs_diff_neg_self_l947_94780

theorem abs_diff_neg_self (m : ℝ) (h : m < 0) : |m - (-m)| = -2*m := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_neg_self_l947_94780


namespace NUMINAMATH_CALUDE_projection_of_congruent_vectors_l947_94716

/-- Definition of vector congruence -/
def is_congruent (a b : ℝ × ℝ) : Prop :=
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.sqrt (a.1^2 + a.2^2) / Real.sqrt (b.1^2 + b.2^2) = Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

/-- Theorem: Projection of a-b on a when b is congruent to a -/
theorem projection_of_congruent_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0))
    (h_congruent : is_congruent b a) :
  let proj := ((a.1 - b.1) * a.1 + (a.2 - b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  proj = (a.1^2 + a.2^2 - (b.1^2 + b.2^2)) / Real.sqrt (a.1^2 + a.2^2) := by
  sorry

end NUMINAMATH_CALUDE_projection_of_congruent_vectors_l947_94716


namespace NUMINAMATH_CALUDE_line_intersects_circle_l947_94762

/-- The line l with equation x - ky - 1 = 0 intersects the circle C with equation x^2 + y^2 = 2 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), (x - k*y - 1 = 0) ∧ (x^2 + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l947_94762


namespace NUMINAMATH_CALUDE_marked_box_second_row_l947_94777

/-- Represents the number of cakes in each box of the pyramid -/
structure CakePyramid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ

/-- The condition that each box in a higher row contains the sum of cakes in the two adjacent boxes below -/
def valid_pyramid (p : CakePyramid) : Prop :=
  p.e = p.a + p.b ∧
  p.f = p.b + p.c ∧
  p.g = p.c + p.d ∧
  p.h = p.e + p.f

/-- The condition that three boxes contain 3, 5, and 6 cakes -/
def marked_boxes (p : CakePyramid) : Prop :=
  (p.a = 3 ∨ p.a = 5 ∨ p.a = 6) ∧
  (p.d = 3 ∨ p.d = 5 ∨ p.d = 6) ∧
  (p.f = 3 ∨ p.f = 5 ∨ p.f = 6) ∧
  p.a ≠ p.d ∧ p.a ≠ p.f ∧ p.d ≠ p.f

/-- The total number of cakes in the pyramid -/
def total_cakes (p : CakePyramid) : ℕ :=
  p.a + p.b + p.c + p.d + p.e + p.f + p.g + p.h

/-- The theorem stating that the marked box in the second row from the bottom contains 3 cakes -/
theorem marked_box_second_row (p : CakePyramid) :
  valid_pyramid p → marked_boxes p → (∀ q : CakePyramid, valid_pyramid q → marked_boxes q → total_cakes q ≥ total_cakes p) → p.f = 3 := by
  sorry


end NUMINAMATH_CALUDE_marked_box_second_row_l947_94777


namespace NUMINAMATH_CALUDE_infinitely_many_composite_generating_numbers_l947_94704

theorem infinitely_many_composite_generating_numbers :
  ∃ f : ℕ → ℕ, Infinite {k | ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + f k = x * y} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_generating_numbers_l947_94704


namespace NUMINAMATH_CALUDE_journey_distance_l947_94747

/-- Proves that a journey with given conditions results in a total distance of 560 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ) 
  (h1 : total_time = 25)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  let total_distance := total_time * (speed_first_half + speed_second_half) / 2
  total_distance = 560 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l947_94747


namespace NUMINAMATH_CALUDE_similar_triangle_area_reduction_l947_94713

/-- Given a right-angled triangle with area A and hypotenuse H, if a smaller similar triangle
    is formed by cutting parallel to the hypotenuse such that the new hypotenuse H' = 0.65H,
    then the area A' of the smaller triangle is equal to A * (0.65)^2. -/
theorem similar_triangle_area_reduction (A H H' A' : ℝ) 
    (h1 : A > 0) 
    (h2 : H > 0) 
    (h3 : H' = 0.65 * H) 
    (h4 : A' / A = (H' / H)^2) : 
  A' = A * (0.65)^2 := by
  sorry

#check similar_triangle_area_reduction

end NUMINAMATH_CALUDE_similar_triangle_area_reduction_l947_94713


namespace NUMINAMATH_CALUDE_midsphere_radius_is_geometric_mean_l947_94731

/-- A regular tetrahedron with its associated spheres -/
structure RegularTetrahedron where
  /-- The radius of the insphere (inscribed sphere) -/
  r_in : ℝ
  /-- The radius of the circumsphere (circumscribed sphere) -/
  r_out : ℝ
  /-- The radius of the midsphere (edge-touching sphere) -/
  r_mid : ℝ
  /-- The radii are positive -/
  h_positive : r_in > 0 ∧ r_out > 0 ∧ r_mid > 0

/-- The radius of the midsphere is the geometric mean of the radii of the insphere and circumsphere -/
theorem midsphere_radius_is_geometric_mean (t : RegularTetrahedron) :
  t.r_mid ^ 2 = t.r_in * t.r_out := by
  sorry

end NUMINAMATH_CALUDE_midsphere_radius_is_geometric_mean_l947_94731


namespace NUMINAMATH_CALUDE_min_boxes_for_load_l947_94773

theorem min_boxes_for_load (total_load : ℝ) (max_box_weight : ℝ) : 
  total_load = 13.5 * 1000 → 
  max_box_weight = 350 → 
  ⌈total_load / max_box_weight⌉ ≥ 39 := by
sorry

end NUMINAMATH_CALUDE_min_boxes_for_load_l947_94773


namespace NUMINAMATH_CALUDE_not_right_triangle_l947_94793

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end NUMINAMATH_CALUDE_not_right_triangle_l947_94793


namespace NUMINAMATH_CALUDE_juniors_score_l947_94776

/-- Given a class with juniors and seniors, prove the juniors' score -/
theorem juniors_score (n : ℕ) (junior_score : ℝ) :
  n > 0 →
  (0.2 * n : ℝ) * junior_score + (0.8 * n : ℝ) * 80 = n * 82 →
  junior_score = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_juniors_score_l947_94776


namespace NUMINAMATH_CALUDE_vector_subtraction_l947_94746

theorem vector_subtraction (u v : Fin 3 → ℝ) 
  (hu : u = ![-3, 5, 2]) 
  (hv : v = ![1, -1, 3]) : 
  u - 2 • v = ![-5, 7, -4] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l947_94746


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l947_94734

/-- The distance from the right focus of the hyperbola x²/4 - y² = 1 to its asymptote x - 2y = 0 is 1 -/
theorem distance_focus_to_asymptote (x y : ℝ) : 
  let hyperbola := (x^2 / 4 - y^2 = 1)
  let right_focus := (x = Real.sqrt 5 ∧ y = 0)
  let asymptote := (x - 2*y = 0)
  let distance := |x - 2*y| / Real.sqrt 5
  (hyperbola ∧ right_focus ∧ asymptote) → distance = 1 := by
sorry


end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l947_94734


namespace NUMINAMATH_CALUDE_parking_fines_count_l947_94723

/-- Represents the number of citations issued for each category -/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  parking : ℕ

/-- Theorem stating that given the conditions, the number of parking fines is 16 -/
theorem parking_fines_count (c : Citations) : 
  c.littering = 4 ∧ 
  c.littering = c.offLeash ∧ 
  c.littering + c.offLeash + c.parking = 24 → 
  c.parking = 16 := by
sorry

end NUMINAMATH_CALUDE_parking_fines_count_l947_94723


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l947_94732

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := ∃ m : ℝ, 1 < m ∧ m < 2 ∧ x = (1/2)^(m-1)

-- Part I
theorem range_of_x_when_a_is_quarter :
  ∀ x : ℝ, (p x (1/4) ∧ q x) ↔ (1/2 < x ∧ x < 3/4) :=
sorry

-- Part II
theorem range_of_a_when_q_sufficient_not_necessary :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ↔ 
  (∀ a : ℝ, (1/3 ≤ a ∧ a ≤ 1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l947_94732


namespace NUMINAMATH_CALUDE_fraction_division_equality_l947_94794

theorem fraction_division_equality : (3 / 8) / (5 / 9) = 27 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l947_94794


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l947_94764

/-- A scaling transformation in a 2D plane -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Apply a scaling transformation to a point -/
def applyTransformation (t : ScalingTransformation) (p : Point) : Point :=
  { x := t.x_scale * p.x,
    y := t.y_scale * p.y }

theorem scaling_transformation_result :
  let A : Point := { x := 1/3, y := -2 }
  let φ : ScalingTransformation := { x_scale := 3, y_scale := 1/2 }
  let A' : Point := applyTransformation φ A
  A'.x = 1 ∧ A'.y = -1 := by sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l947_94764


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l947_94761

def chairs_per_row : ℕ := 13
def initial_chairs : ℕ := 169
def expected_attendees : ℕ := 95
def max_removable_chairs : ℕ := 26

theorem optimal_chair_removal :
  ∀ n : ℕ,
  n ≤ max_removable_chairs →
  (initial_chairs - n) % chairs_per_row = 0 →
  initial_chairs - max_removable_chairs ≤ initial_chairs - n →
  (initial_chairs - n) - expected_attendees ≥
    (initial_chairs - max_removable_chairs) - expected_attendees :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l947_94761


namespace NUMINAMATH_CALUDE_hyperbola_equation_l947_94738

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance
  e : ℝ  -- eccentricity

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem: For a hyperbola with given properties, prove its standard equation -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_b : h.b = 12)
  (h_e : h.e = 5/4)
  (h_foci : h.c^2 = h.a^2 + h.b^2)
  (x y : ℝ) :
  standard_equation h x y ↔ x^2 / 64 - y^2 / 36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l947_94738


namespace NUMINAMATH_CALUDE_tunnel_length_l947_94792

/-- The length of a tunnel given a train passing through it. -/
theorem tunnel_length
  (train_length : ℝ)
  (transit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : transit_time = 4 / 60)  -- 4 minutes converted to hours
  (h3 : train_speed = 90) :
  train_speed * transit_time - train_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_l947_94792


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l947_94711

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 34) : 
  a * b = 4.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l947_94711


namespace NUMINAMATH_CALUDE_max_white_pieces_l947_94797

/-- Represents the color of a piece -/
inductive Color
| Black
| White

/-- Represents the circle of pieces -/
def Circle := List Color

/-- The initial configuration of the circle -/
def initial_circle : Circle :=
  [Color.Black, Color.Black, Color.Black, Color.Black, Color.White]

/-- Applies the rules to place new pieces and remove old ones -/
def apply_rules (c : Circle) : Circle :=
  sorry

/-- Counts the number of white pieces in the circle -/
def count_white (c : Circle) : Nat :=
  sorry

/-- Theorem stating that the maximum number of white pieces is 3 -/
theorem max_white_pieces (c : Circle) : 
  count_white (apply_rules c) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_white_pieces_l947_94797


namespace NUMINAMATH_CALUDE_ticket_price_ratio_l947_94742

/-- Proves that the ratio of adult to child ticket prices is 2:1 given the problem conditions --/
theorem ticket_price_ratio :
  ∀ (adult_price child_price : ℚ),
    adult_price = 32 →
    400 * adult_price + 200 * child_price = 16000 →
    adult_price / child_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_ratio_l947_94742


namespace NUMINAMATH_CALUDE_shopping_trip_remainder_l947_94768

/-- Calculates the remaining amount after a shopping trip --/
theorem shopping_trip_remainder
  (initial_amount : ℝ)
  (peach_price peach_quantity : ℝ)
  (cherry_price cherry_quantity : ℝ)
  (baguette_price baguette_quantity : ℝ)
  (discount_threshold discount_rate : ℝ)
  (tax_rate : ℝ)
  (h1 : initial_amount = 20)
  (h2 : peach_price = 2)
  (h3 : peach_quantity = 3)
  (h4 : cherry_price = 3.5)
  (h5 : cherry_quantity = 2)
  (h6 : baguette_price = 1.25)
  (h7 : baguette_quantity = 4)
  (h8 : discount_threshold = 10)
  (h9 : discount_rate = 0.1)
  (h10 : tax_rate = 0.05) :
  let subtotal := peach_price * peach_quantity + cherry_price * cherry_quantity + baguette_price * baguette_quantity
  let discounted_total := if subtotal > discount_threshold then subtotal * (1 - discount_rate) else subtotal
  let final_total := discounted_total * (1 + tax_rate)
  let remainder := initial_amount - final_total
  remainder = 2.99 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_remainder_l947_94768


namespace NUMINAMATH_CALUDE_a_range_l947_94706

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

theorem a_range (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l947_94706


namespace NUMINAMATH_CALUDE_cookie_distribution_l947_94759

/-- Cookie distribution problem -/
theorem cookie_distribution (b m l : ℕ) : 
  b + m + l = 30 ∧ 
  m = 2 * b ∧ 
  l = b + m → 
  b = 5 ∧ m = 10 ∧ l = 15 := by
  sorry

#check cookie_distribution

end NUMINAMATH_CALUDE_cookie_distribution_l947_94759


namespace NUMINAMATH_CALUDE_real_part_of_complex_power_l947_94736

theorem real_part_of_complex_power : Complex.re ((1 - 2*Complex.I)^5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_power_l947_94736


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l947_94701

/-- The minimum distance between a point on the parabola y^2 = 6x and a point on the circle (x-4)^2 + y^2 = 1 is √15 - 1 -/
theorem min_distance_parabola_circle :
  ∃ (d : ℝ), d = Real.sqrt 15 - 1 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁^2 = 6*x₁ →
    (x₂ - 4)^2 + y₂^2 = 1 →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 : ℝ) ≥ d^2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l947_94701


namespace NUMINAMATH_CALUDE_compote_level_reduction_l947_94737

theorem compote_level_reduction (V : ℝ) (h : V > 0) :
  let initial_level := V
  let level_after_third := 3/4 * V
  let volume_of_remaining_peaches := 1/6 * V
  let final_level := level_after_third - volume_of_remaining_peaches
  (level_after_third - final_level) / level_after_third = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_compote_level_reduction_l947_94737


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l947_94740

/-- Simple interest calculation for a given principal, rate, and interest amount -/
theorem simple_interest_time_calculation 
  (P : ℝ) (R : ℝ) (SI : ℝ) (h1 : P = 10000) (h2 : R = 5) (h3 : SI = 500) : 
  (SI * 100) / (P * R) * 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l947_94740


namespace NUMINAMATH_CALUDE_chef_potato_problem_l947_94705

/-- The number of potatoes already cooked -/
def potatoes_cooked : ℕ := 7

/-- The time it takes to cook one potato (in minutes) -/
def cooking_time_per_potato : ℕ := 5

/-- The time it takes to cook the remaining potatoes (in minutes) -/
def remaining_cooking_time : ℕ := 45

/-- The total number of potatoes the chef needs to cook -/
def total_potatoes : ℕ := 16

theorem chef_potato_problem :
  total_potatoes = potatoes_cooked + remaining_cooking_time / cooking_time_per_potato :=
by sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l947_94705


namespace NUMINAMATH_CALUDE_tan_double_angle_l947_94703

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l947_94703


namespace NUMINAMATH_CALUDE_fraction_subtraction_l947_94751

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l947_94751


namespace NUMINAMATH_CALUDE_max_min_sum_l947_94757

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum (M m : ℝ) (hM : ∀ x, f x ≤ M) (hm : ∀ x, m ≤ f x) : M + m = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_l947_94757


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l947_94712

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence
  (a : ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d 3 = 23)
  (h2 : arithmetic_sequence a d 7 = 35) :
  arithmetic_sequence a d 10 = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l947_94712


namespace NUMINAMATH_CALUDE_land_area_needed_l947_94753

def land_cost_per_sqm : ℝ := 50
def brick_cost_per_1000 : ℝ := 100
def roof_tile_cost_per_tile : ℝ := 10
def num_bricks_needed : ℝ := 10000
def num_roof_tiles_needed : ℝ := 500
def total_construction_cost : ℝ := 106000

theorem land_area_needed :
  ∃ (x : ℝ),
    x * land_cost_per_sqm +
    (num_bricks_needed / 1000) * brick_cost_per_1000 +
    num_roof_tiles_needed * roof_tile_cost_per_tile =
    total_construction_cost ∧
    x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_land_area_needed_l947_94753


namespace NUMINAMATH_CALUDE_tank_capacity_l947_94756

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (used_gallons : ℕ) : 
  initial_fraction = 3/4 → 
  final_fraction = 1/4 → 
  used_gallons = 24 → 
  ∃ (total_capacity : ℕ), 
    total_capacity = 48 ∧ 
    (initial_fraction - final_fraction) * total_capacity = used_gallons :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l947_94756


namespace NUMINAMATH_CALUDE_problem_solution_l947_94748

theorem problem_solution (m n : ℝ) (h1 : m - n = 6) (h2 : m * n = 4) : 
  (m^2 + n^2 = 44) ∧ ((m + 2) * (n - 2) = -12) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l947_94748


namespace NUMINAMATH_CALUDE_seventh_roots_of_unity_polynomial_factorization_l947_94766

theorem seventh_roots_of_unity_polynomial_factorization (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b₁*x + c₁)*(x^2 + b₂*x + c₂)*(x^2 + b₃*x + c₃)) :
  b₁*c₁ + b₂*c₂ + b₃*c₃ = 1 := by
sorry

end NUMINAMATH_CALUDE_seventh_roots_of_unity_polynomial_factorization_l947_94766


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l947_94715

def vector_a : Fin 2 → ℝ := ![(-2), 3]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![3, m]

theorem perpendicular_vectors (m : ℝ) :
  (vector_a 0 * vector_b m 0 + vector_a 1 * vector_b m 1 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l947_94715


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l947_94708

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Counts the number of adjacent pairs with the same color -/
def countSameColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of adjacent pairs with different colors -/
def countDifferentColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the equal pairs condition -/
def isValidArrangement (arr : Arrangement) : Prop :=
  countSameColorPairs arr = countDifferentColorPairs arr

/-- Counts the number of blue marbles in an arrangement -/
def countBlueMarbles (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of yellow marbles in an arrangement -/
def countYellowMarbles (arr : Arrangement) : Nat :=
  sorry

theorem marble_arrangement_theorem :
  ∃ (validArrangements : List Arrangement),
    (∀ arr ∈ validArrangements, isValidArrangement arr) ∧
    (∀ arr ∈ validArrangements, countBlueMarbles arr = 4) ∧
    (∀ arr ∈ validArrangements, countYellowMarbles arr ≤ 11) ∧
    (validArrangements.length = 35) :=
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l947_94708


namespace NUMINAMATH_CALUDE_rajan_share_is_2400_l947_94729

/-- Calculates the share of profit for a partner in a business based on investments and durations. -/
def calculate_share (rajan_investment : ℕ) (rajan_duration : ℕ) 
                    (rakesh_investment : ℕ) (rakesh_duration : ℕ)
                    (mukesh_investment : ℕ) (mukesh_duration : ℕ)
                    (total_profit : ℕ) : ℕ :=
  let rajan_product := rajan_investment * rajan_duration
  let rakesh_product := rakesh_investment * rakesh_duration
  let mukesh_product := mukesh_investment * mukesh_duration
  let total_product := rajan_product + rakesh_product + mukesh_product
  (rajan_product * total_profit) / total_product

/-- Theorem stating that Rajan's share of the profit is 2400 given the specified investments and durations. -/
theorem rajan_share_is_2400 :
  calculate_share 20000 12 25000 4 15000 8 4600 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_rajan_share_is_2400_l947_94729


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l947_94760

/-- Given a hyperbola and a parabola with specific properties, prove that the parameter p of the parabola equals 2 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : p > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∀ x y : ℝ, y^2 = 2*p*x) →               -- Parabola equation
  (b / a = Real.sqrt 3) →                  -- Derived from eccentricity = 2
  (p^2 * Real.sqrt 3 / 4 = Real.sqrt 3) →  -- Derived from area of triangle AOB
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l947_94760


namespace NUMINAMATH_CALUDE_forum_questions_per_hour_l947_94788

/-- Proves that given the conditions of the forum, the average number of questions posted by each user per hour is 3 -/
theorem forum_questions_per_hour (members : ℕ) (total_posts_per_day : ℕ) 
  (h1 : members = 200)
  (h2 : total_posts_per_day = 57600) : 
  (total_posts_per_day / (24 * members)) / 4 = 3 := by
  sorry

#check forum_questions_per_hour

end NUMINAMATH_CALUDE_forum_questions_per_hour_l947_94788


namespace NUMINAMATH_CALUDE_almond_butter_cookie_cost_difference_l947_94771

/-- The cost difference per batch between almond butter cookies and peanut butter cookies -/
def cost_difference (peanut_butter_cost : ℚ) (almond_butter_multiplier : ℚ) 
  (jar_fraction : ℚ) (sugar_difference : ℚ) : ℚ :=
  let almond_butter_cost := peanut_butter_cost * almond_butter_multiplier
  let peanut_butter_batch_cost := peanut_butter_cost * jar_fraction
  let almond_butter_batch_cost := almond_butter_cost * jar_fraction
  (almond_butter_batch_cost - peanut_butter_batch_cost) + sugar_difference

/-- Theorem stating the additional cost per batch for almond butter cookies -/
theorem almond_butter_cookie_cost_difference : 
  cost_difference 3 3 (1/2) (1/2) = 7/2 := by
  sorry

#eval cost_difference 3 3 (1/2) (1/2)

end NUMINAMATH_CALUDE_almond_butter_cookie_cost_difference_l947_94771


namespace NUMINAMATH_CALUDE_fraction_comparison_l947_94754

theorem fraction_comparison (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5/3 → 
  (8*x - 3 > 5 - 3*x ↔ (8/11 < x ∧ x < 5/3) ∨ (5/3 < x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l947_94754


namespace NUMINAMATH_CALUDE_filter_kit_price_calculation_l947_94796

/-- The price of a camera lens filter kit -/
def filter_kit_price (price1 price2 price3 : ℝ) (discount : ℝ) : ℝ :=
  let total_individual := 2 * price1 + 2 * price2 + price3
  total_individual * (1 - discount)

/-- Theorem stating the price of the filter kit -/
theorem filter_kit_price_calculation :
  filter_kit_price 16.45 14.05 19.50 0.08 = 74.06 := by
  sorry

end NUMINAMATH_CALUDE_filter_kit_price_calculation_l947_94796


namespace NUMINAMATH_CALUDE_f_definition_l947_94791

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_definition (x : ℝ) : f x = 2 - x :=
  sorry

end NUMINAMATH_CALUDE_f_definition_l947_94791


namespace NUMINAMATH_CALUDE_frank_skee_ball_tickets_l947_94709

/-- The number of tickets Frank won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 33

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 6

/-- The number of candies Frank can buy with his total tickets -/
def candies_bought : ℕ := 7

/-- The number of tickets Frank won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets : skee_ball_tickets = 9 := by
  sorry

end NUMINAMATH_CALUDE_frank_skee_ball_tickets_l947_94709


namespace NUMINAMATH_CALUDE_circle_equation_through_pole_l947_94721

/-- A circle in a polar coordinate system --/
structure PolarCircle where
  center : (ℝ × ℝ)
  passes_through_pole : Bool

/-- The polar coordinate equation of a circle --/
def polar_equation (c : PolarCircle) : ℝ → ℝ := sorry

theorem circle_equation_through_pole (c : PolarCircle) 
  (h1 : c.center = (Real.sqrt 2, Real.pi))
  (h2 : c.passes_through_pole = true) :
  polar_equation c = λ θ => -2 * Real.sqrt 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_pole_l947_94721


namespace NUMINAMATH_CALUDE_difference_not_necessarily_periodic_l947_94770

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ 
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x)

-- Define functions g and h with their respective periods
def g_periodic (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 6) = g x

def h_periodic (h : ℝ → ℝ) : Prop :=
  ∀ x, h (x + 2 * Real.pi) = h x

-- Theorem statement
theorem difference_not_necessarily_periodic 
  (g h : ℝ → ℝ) 
  (hg : g_periodic g) 
  (hh : h_periodic h) :
  ¬ (∀ f : ℝ → ℝ, f = g - h → Periodic f) :=
sorry

end NUMINAMATH_CALUDE_difference_not_necessarily_periodic_l947_94770


namespace NUMINAMATH_CALUDE_printer_X_time_l947_94707

/-- The time (in hours) it takes for printer Y to complete the job alone -/
def time_Y : ℝ := 12

/-- The time (in hours) it takes for printer Z to complete the job alone -/
def time_Z : ℝ := 8

/-- The ratio of the time it takes printer X alone to the time it takes printers Y and Z together -/
def ratio : ℝ := 3.333333333333333

theorem printer_X_time : ∃ (time_X : ℝ), time_X = 16 ∧
  ratio = time_X / (1 / (1 / time_Y + 1 / time_Z)) :=
sorry

end NUMINAMATH_CALUDE_printer_X_time_l947_94707


namespace NUMINAMATH_CALUDE_solution_set_inequality_range_of_m_l947_94727

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 :=
sorry

-- Theorem for part 2
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - f (-x) ≤ 4 / a + 1 / b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_range_of_m_l947_94727


namespace NUMINAMATH_CALUDE_solve_for_A_l947_94722

theorem solve_for_A : ∃ A : ℚ, 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 ∧ A = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l947_94722


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l947_94720

theorem quadratic_solution_range (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  (f 3.24 = -0.02) → (f 3.25 = 0.01) → (f 3.26 = 0.03) →
  ∃ x, f x = 0 ∧ 3.24 < x ∧ x < 3.25 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l947_94720


namespace NUMINAMATH_CALUDE_inverse_matrices_l947_94726

/-- Two 2x2 matrices are inverses if their product is the identity matrix -/
def are_inverses (A B : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * B = !![1, 0; 0, 1]

/-- Matrix A definition -/
def A (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![x, 3; 1, 5]

/-- Matrix B definition -/
def B (y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![-5/31, 1/31; y, 3/31]

/-- The theorem to be proved -/
theorem inverse_matrices :
  are_inverses (A (-9)) (B (1/31)) := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_l947_94726


namespace NUMINAMATH_CALUDE_max_cables_used_l947_94714

/-- Represents the number of brand A computers -/
def brand_A_count : Nat := 25

/-- Represents the number of brand B computers -/
def brand_B_count : Nat := 15

/-- Represents the total number of employees -/
def total_employees : Nat := brand_A_count + brand_B_count

/-- Theorem stating the maximum number of cables that can be used -/
theorem max_cables_used : 
  ∀ (cables : Nat), 
    (cables ≤ brand_A_count * brand_B_count) → 
    (∀ (b : Nat), b < brand_B_count → ∃ (a : Nat), a < brand_A_count ∧ True) → 
    cables ≤ 375 :=
by sorry

end NUMINAMATH_CALUDE_max_cables_used_l947_94714


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l947_94781

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l947_94781


namespace NUMINAMATH_CALUDE_ramanujan_hardy_game_l947_94779

theorem ramanujan_hardy_game (h r : ℂ) : 
  h * r = 32 - 8 * I ∧ h = 5 + 3 * I → r = 4 - 4 * I := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_game_l947_94779


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l947_94775

/-- Two lines L₁ and L₂ are defined as follows:
    L₁: ax + 3y + 1 = 0
    L₂: 2x + (a+1)y + 1 = 0
    This theorem proves that if L₁ and L₂ are parallel, then a = -3. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 3*y + 1 = 0 ↔ 2*x + (a+1)*y + 1 = 0) →
  a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l947_94775


namespace NUMINAMATH_CALUDE_fan_work_time_theorem_l947_94743

/-- Represents the fan's properties and operation --/
structure Fan where
  airflow_rate : ℝ  -- liters per second
  work_time : ℝ     -- minutes per day
  total_airflow : ℝ -- liters per week

/-- Theorem stating the relationship between fan operation and total airflow --/
theorem fan_work_time_theorem (f : Fan) (h1 : f.airflow_rate = 10) 
  (h2 : f.total_airflow = 42000) : 
  f.work_time = 10 ↔ f.total_airflow = 7 * f.work_time * 60 * f.airflow_rate := by
  sorry

#check fan_work_time_theorem

end NUMINAMATH_CALUDE_fan_work_time_theorem_l947_94743


namespace NUMINAMATH_CALUDE_complex_sum_equality_l947_94767

theorem complex_sum_equality : 
  5 * Complex.exp (Complex.I * Real.pi / 12) + 5 * Complex.exp (Complex.I * 13 * Real.pi / 24) 
  = 10 * Real.cos (11 * Real.pi / 48) * Complex.exp (Complex.I * 5 * Real.pi / 16) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l947_94767


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l947_94785

theorem fraction_sum_equals_two (a b : ℝ) (h : a ≠ b) : 
  (2 * a) / (a - b) + (2 * b) / (b - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l947_94785


namespace NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l947_94774

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by sorry

end NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l947_94774


namespace NUMINAMATH_CALUDE_sum_of_products_l947_94739

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 36) :
  x*y + y*z + x*z = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l947_94739


namespace NUMINAMATH_CALUDE_hotel_guests_count_l947_94717

/-- The number of guests attending at least one reunion -/
def total_guests (oates_guests hall_guests both_guests : ℕ) : ℕ :=
  oates_guests + hall_guests - both_guests

/-- Theorem stating the total number of guests attending at least one reunion -/
theorem hotel_guests_count :
  total_guests 70 52 28 = 94 := by
  sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l947_94717


namespace NUMINAMATH_CALUDE_initial_lives_proof_l947_94784

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb lost -/
def lives_lost : ℕ := 25

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the initial number of lives equals the sum of remaining lives and lives lost -/
theorem initial_lives_proof : initial_lives = remaining_lives + lives_lost := by
  sorry

end NUMINAMATH_CALUDE_initial_lives_proof_l947_94784


namespace NUMINAMATH_CALUDE_sphere_only_orientation_independent_l947_94795

-- Define the types of 3D objects we're considering
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RegularTriangularPyramid
  | Sphere

-- Define a function that determines if an object's orthographic projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_orientation_independent :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_orientation_independent_l947_94795


namespace NUMINAMATH_CALUDE_min_value_expression_l947_94702

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 48) :
  x^2 + 6*x*y + 9*y^2 + 4*z^2 ≥ 128 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 6*x₀*y₀ + 9*y₀^2 + 4*z₀^2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l947_94702


namespace NUMINAMATH_CALUDE_parabola_locus_l947_94782

/-- The locus of points from which a parabola is seen at a 45° angle -/
theorem parabola_locus (p : ℝ) (u v : ℝ) : 
  (∃ (m₁ m₂ : ℝ), 
    -- Two distinct tangent lines exist
    m₁ ≠ m₂ ∧
    -- The tangent lines touch the parabola
    (∀ (x y : ℝ), y^2 = 2*p*x → (y - v = m₁*(x - u) ∨ y - v = m₂*(x - u))) ∧
    -- The angle between the tangent lines is 45°
    (m₁ - m₂) / (1 + m₁*m₂) = 1) →
  -- The point (u, v) lies on the hyperbola
  (u + 3*p/2)^2 - v^2 = 2*p^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_locus_l947_94782


namespace NUMINAMATH_CALUDE_fifteen_percent_greater_l947_94700

theorem fifteen_percent_greater : ∃ (x : ℝ), (15 / 100 * 40 = 25 / 100 * x + 2) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_greater_l947_94700


namespace NUMINAMATH_CALUDE_valid_tiling_exists_l947_94787

/-- Represents a point on the infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a domino piece on the grid -/
inductive Domino
  | Horizontal (topLeft : GridPoint)
  | Vertical (topLeft : GridPoint)

/-- Represents a tiling of the infinite grid with dominos -/
def Tiling := GridPoint → Domino

/-- Checks if a given point is covered by a domino in the tiling -/
def isCovered (t : Tiling) (p : GridPoint) : Prop := 
  ∃ d : Domino, d ∈ Set.range t ∧ 
    match d with
    | Domino.Horizontal tl => p.x = tl.x ∧ (p.y = tl.y ∨ p.y = tl.y + 1)
    | Domino.Vertical tl => p.y = tl.y ∧ (p.x = tl.x ∨ p.x = tl.x + 1)

/-- Checks if a horizontal line intersects a finite number of dominos -/
def finiteHorizontalIntersections (t : Tiling) : Prop :=
  ∀ y : ℤ, ∃ n : ℕ, ∀ x : ℤ, x > n → 
    (t ⟨x, y⟩ = t ⟨x - 1, y⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- Checks if a vertical line intersects a finite number of dominos -/
def finiteVerticalIntersections (t : Tiling) : Prop :=
  ∀ x : ℤ, ∃ n : ℕ, ∀ y : ℤ, y > n → 
    (t ⟨x, y⟩ = t ⟨x, y - 1⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- The main theorem stating that a valid tiling with the required properties exists -/
theorem valid_tiling_exists : 
  ∃ t : Tiling, 
    (∀ p : GridPoint, isCovered t p) ∧ 
    finiteHorizontalIntersections t ∧
    finiteVerticalIntersections t := by
  sorry

end NUMINAMATH_CALUDE_valid_tiling_exists_l947_94787


namespace NUMINAMATH_CALUDE_watermelon_total_sold_l947_94799

def watermelon_problem (customers_one : Nat) (customers_three : Nat) (customers_two : Nat) : Nat :=
  customers_one * 1 + customers_three * 3 + customers_two * 2

theorem watermelon_total_sold :
  watermelon_problem 17 3 10 = 46 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_total_sold_l947_94799


namespace NUMINAMATH_CALUDE_teacher_health_survey_l947_94741

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 80)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 100/3 :=
sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l947_94741


namespace NUMINAMATH_CALUDE_john_scores_42_points_l947_94725

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (points_per_interval : ℕ) (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_duration := num_periods * period_duration
  let num_intervals := total_duration / interval_duration
  points_per_interval * num_intervals

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  let points_per_interval := 2 * 2 + 1 * 3  -- 2 two-point shots and 1 three-point shot
  let interval_duration := 4                -- every 4 minutes
  let num_periods := 2                      -- 2 periods
  let period_duration := 12                 -- each period is 12 minutes
  total_points_scored points_per_interval interval_duration num_periods period_duration = 42 := by
  sorry


end NUMINAMATH_CALUDE_john_scores_42_points_l947_94725


namespace NUMINAMATH_CALUDE_reciprocal_problem_l947_94778

theorem reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l947_94778


namespace NUMINAMATH_CALUDE_negative_two_x_times_three_y_l947_94763

theorem negative_two_x_times_three_y (x y : ℝ) : -2 * x * 3 * y = -6 * x * y := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_times_three_y_l947_94763


namespace NUMINAMATH_CALUDE_complex_number_properties_l947_94749

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m + 2) (m^2 - 1)

theorem complex_number_properties :
  (∀ m : ℝ, z m = 0 ↔ m = 1) ∧
  (∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2) ∧
  (∀ m : ℝ, (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l947_94749


namespace NUMINAMATH_CALUDE_problem_solution_l947_94752

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l947_94752


namespace NUMINAMATH_CALUDE_acid_dilution_l947_94718

/-- Proves that adding 80 ounces of pure water to 50 ounces of a 26% acid solution
    results in a 10% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.26 →
  water_added = 80 →
  final_concentration = 0.10 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l947_94718


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l947_94735

/-- Calculates the number of pants after a given number of years -/
def pantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * (pairsPerYear * pantsPerPair)

/-- Theorem: Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  pantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval pantsAfterYears 50 4 2 5

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l947_94735


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l947_94783

theorem quadratic_equations_common_root (a b c x : ℝ) 
  (h1 : a * c ≠ 0) (h2 : a ≠ c) 
  (hM : a * x^2 + b * x + c = 0) 
  (hN : c * x^2 + b * x + a = 0) : 
  x = 1 ∨ x = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l947_94783


namespace NUMINAMATH_CALUDE_tan_4050_undefined_l947_94750

theorem tan_4050_undefined :
  ∀ x : ℝ, Real.tan (4050 * π / 180) = x → False :=
by
  sorry

end NUMINAMATH_CALUDE_tan_4050_undefined_l947_94750
