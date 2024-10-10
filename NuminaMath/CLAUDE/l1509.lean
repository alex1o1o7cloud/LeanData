import Mathlib

namespace comic_book_pages_l1509_150944

theorem comic_book_pages (total_frames : Nat) (frames_per_page : Nat) 
  (h1 : total_frames = 143)
  (h2 : frames_per_page = 11) :
  (total_frames / frames_per_page = 13) ∧ (total_frames % frames_per_page = 0) := by
  sorry

end comic_book_pages_l1509_150944


namespace rectangle_shorter_side_l1509_150948

theorem rectangle_shorter_side 
  (width : Real) 
  (num_poles : Nat) 
  (pole_distance : Real) 
  (h1 : width = 50) 
  (h2 : num_poles = 24) 
  (h3 : pole_distance = 5) : 
  ∃ length : Real, 
    length = 7.5 ∧ 
    length ≤ width ∧ 
    2 * (length + width) = (num_poles - 1 : Real) * pole_distance := by
  sorry

end rectangle_shorter_side_l1509_150948


namespace third_grade_swim_caps_l1509_150991

theorem third_grade_swim_caps (b r : ℕ) : 
  b = 4 * r + 2 →
  b = r + 24 →
  b + r = 37 :=
by sorry

end third_grade_swim_caps_l1509_150991


namespace exam_average_l1509_150962

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 70 / 100 →
  avg_total = 80 / 100 →
  ∃ avg₂ : ℚ, 
    (n₁.cast * avg₁ + n₂.cast * avg₂) / (n₁ + n₂).cast = avg_total ∧
    avg₂ = 95 / 100 :=
by sorry

end exam_average_l1509_150962


namespace greatest_consecutive_integers_sum_72_l1509_150919

/-- The sum of N consecutive integers starting from a -/
def sumConsecutiveIntegers (N : ℕ) (a : ℤ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 144 is the greatest number of consecutive integers summing to 72 -/
theorem greatest_consecutive_integers_sum_72 :
  ∀ N : ℕ, (∃ a : ℤ, sumConsecutiveIntegers N a = 72) → N ≤ 144 :=
by sorry

end greatest_consecutive_integers_sum_72_l1509_150919


namespace basic_astrophysics_degrees_l1509_150963

def microphotonics : ℝ := 13
def home_electronics : ℝ := 24
def food_additives : ℝ := 15
def genetically_modified_microorganisms : ℝ := 29
def industrial_lubricants : ℝ := 8
def total_circle_degrees : ℝ := 360

def other_sectors_sum : ℝ := 
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def basic_astrophysics_percentage : ℝ := 100 - other_sectors_sum

theorem basic_astrophysics_degrees : 
  (basic_astrophysics_percentage / 100) * total_circle_degrees = 39.6 := by
  sorry

end basic_astrophysics_degrees_l1509_150963


namespace h_of_h_of_two_equals_91265_l1509_150971

/-- Given a function h(x) = 3x^3 + 2x^2 - x + 1, prove that h(h(2)) = 91265 -/
theorem h_of_h_of_two_equals_91265 : 
  let h : ℝ → ℝ := fun x ↦ 3 * x^3 + 2 * x^2 - x + 1
  h (h 2) = 91265 := by
  sorry

end h_of_h_of_two_equals_91265_l1509_150971


namespace rectangle_area_l1509_150941

theorem rectangle_area (perimeter width length : ℝ) : 
  perimeter = 72 ∧ 
  2 * (length + width) = perimeter ∧ 
  length = 3 * width → 
  length * width = 243 := by
  sorry

end rectangle_area_l1509_150941


namespace symmetry_of_graphs_l1509_150936

theorem symmetry_of_graphs (f : ℝ → ℝ) (a : ℝ) :
  ∀ x y : ℝ, f (a - x) = y ↔ f (x - a) = y :=
sorry

end symmetry_of_graphs_l1509_150936


namespace latest_start_time_l1509_150911

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Represents a turkey roasting scenario -/
structure TurkeyRoast where
  num_turkeys : ℕ
  turkey_weight : ℕ
  roast_time_per_pound : ℕ
  dinner_time : Time

def total_roast_time (tr : TurkeyRoast) : ℕ :=
  tr.num_turkeys * tr.turkey_weight * tr.roast_time_per_pound

def subtract_hours (t : Time) (h : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes - h * 60
  ⟨total_minutes / 60, total_minutes % 60, by sorry⟩

theorem latest_start_time (tr : TurkeyRoast) 
  (h_num : tr.num_turkeys = 2)
  (h_weight : tr.turkey_weight = 16)
  (h_roast_time : tr.roast_time_per_pound = 15)
  (h_dinner : tr.dinner_time = ⟨18, 0, by sorry⟩) :
  subtract_hours tr.dinner_time (total_roast_time tr / 60) = ⟨10, 0, by sorry⟩ :=
sorry

end latest_start_time_l1509_150911


namespace rotten_bananas_percentage_l1509_150945

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 894 / 1000) :
  (total_oranges * (1 - rotten_oranges_percentage) + total_bananas * (1 - (4 / 100 : ℚ))) / (total_oranges + total_bananas) = good_fruits_percentage :=
by sorry

end rotten_bananas_percentage_l1509_150945


namespace four_digit_perfect_cubes_divisible_by_16_l1509_150976

theorem four_digit_perfect_cubes_divisible_by_16 :
  (Finset.filter (fun n : ℕ => 
    1000 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 9999) (Finset.range 1000)).card = 6 := by
  sorry

end four_digit_perfect_cubes_divisible_by_16_l1509_150976


namespace total_pushups_count_l1509_150992

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 44

/-- The additional number of push-ups David did compared to Zachary -/
def david_extra_pushups : ℕ := 58

/-- The total number of push-ups done by Zachary and David -/
def total_pushups : ℕ := zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating the total number of push-ups done by Zachary and David -/
theorem total_pushups_count : total_pushups = 146 := by
  sorry

end total_pushups_count_l1509_150992


namespace masha_comb_teeth_l1509_150993

/-- Represents a comb with teeth --/
structure Comb where
  numTeeth : ℕ
  numGaps : ℕ
  numSegments : ℕ

/-- The relationship between teeth and gaps in a comb --/
axiom comb_structure (c : Comb) : c.numGaps = c.numTeeth - 1

/-- The total number of segments in a comb --/
axiom comb_segments (c : Comb) : c.numSegments = c.numTeeth + c.numGaps

/-- Katya's comb --/
def katya_comb : Comb := { numTeeth := 11, numGaps := 10, numSegments := 21 }

/-- Masha's comb --/
def masha_comb : Comb := { numTeeth := 53, numGaps := 52, numSegments := 105 }

/-- The relationship between Katya's and Masha's combs --/
axiom comb_relationship : masha_comb.numSegments = 5 * katya_comb.numSegments

theorem masha_comb_teeth : masha_comb.numTeeth = 53 := by
  sorry

end masha_comb_teeth_l1509_150993


namespace inscribed_right_triangle_exists_l1509_150932

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

/-- A right triangle -/
structure RightTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point
  is_right_angle : (vertex1.1 - vertex2.1) * (vertex1.1 - vertex3.1) + 
                   (vertex1.2 - vertex2.2) * (vertex1.2 - vertex3.2) = 0

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (c : Circle) (t : RightTriangle) : Prop :=
  let (x1, y1) := t.vertex1
  let (x2, y2) := t.vertex2
  let (x3, y3) := t.vertex3
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x3 - cx)^2 + (y3 - cy)^2 = c.radius^2

/-- Check if a line passes through a point -/
def passesThrough (p1 : Point) (p2 : Point) (p : Point) : Prop :=
  (p.1 - p1.1) * (p2.2 - p1.2) = (p.2 - p1.2) * (p2.1 - p1.1)

theorem inscribed_right_triangle_exists (c : Circle) (A B : Point) 
  (h1 : isInside c A) (h2 : isInside c B) :
  ∃ (t : RightTriangle), isInscribed c t ∧ 
    (passesThrough t.vertex1 t.vertex2 A ∨ passesThrough t.vertex1 t.vertex3 A) ∧
    (passesThrough t.vertex1 t.vertex2 B ∨ passesThrough t.vertex1 t.vertex3 B) := by
  sorry

end inscribed_right_triangle_exists_l1509_150932


namespace similar_triangles_segment_length_l1509_150943

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_segment_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z)
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_YZ : dist Y Z = 24) :
  dist X Y = 12 := by sorry

end similar_triangles_segment_length_l1509_150943


namespace inequality_solution_sets_l1509_150906

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end inequality_solution_sets_l1509_150906


namespace geometric_sequence_second_term_l1509_150980

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 48 and the 6th term is 72, the 2nd term is 1152/81 -/
theorem geometric_sequence_second_term
  (seq : GeometricSequence)
  (h5 : seq.nth_term 5 = 48)
  (h6 : seq.nth_term 6 = 72) :
  seq.nth_term 2 = 1152 / 81 := by
  sorry


end geometric_sequence_second_term_l1509_150980


namespace triangle_problem_l1509_150989

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove that B = π/3 and the maximum area is √3 --/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
  let n : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)
  b = 2 →
  (∃ (k : ℝ), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  B = π / 3 ∧
  (∀ (S : ℝ), S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by sorry

end triangle_problem_l1509_150989


namespace triangle_property_and_function_value_l1509_150977

theorem triangle_property_and_function_value (a b c A : ℝ) :
  0 < A ∧ A < π →
  b^2 + c^2 = a^2 + Real.sqrt 3 * b * c →
  let m : ℝ × ℝ := (Real.sin A, Real.cos A)
  let n : ℝ × ℝ := (Real.cos A, Real.sqrt 3 * Real.cos A)
  let f : ℝ → ℝ := fun x => m.1 * n.1 + m.2 * n.2 - Real.sqrt 3 / 2
  f A = Real.sqrt 3 / 2 := by
sorry

end triangle_property_and_function_value_l1509_150977


namespace number_of_bs_l1509_150934

/-- Represents the number of students who earn each grade in a biology class. -/
structure GradeDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The conditions of the biology class grade distribution. -/
def validGradeDistribution (g : GradeDistribution) : Prop :=
  g.a + g.b + g.c + g.d = 40 ∧
  g.a = 12 * g.b / 10 ∧
  g.c = g.b ∧
  g.d = g.b / 2

/-- The theorem stating that the number of B's in the class is 11. -/
theorem number_of_bs (g : GradeDistribution) 
  (h : validGradeDistribution g) : g.b = 11 := by
  sorry

end number_of_bs_l1509_150934


namespace gcf_of_60_and_75_l1509_150939

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_of_60_and_75_l1509_150939


namespace trigonometric_equation_implications_l1509_150942

theorem trigonometric_equation_implications (x : Real) 
  (h : (Real.sin (Real.pi + x) + 2 * Real.cos (3 * Real.pi / 2 + x)) / 
       (Real.cos (Real.pi - x) - Real.sin (Real.pi / 2 - x)) = 1) : 
  Real.tan x = 2/3 ∧ Real.sin (2*x) - Real.cos x ^ 2 = 3/13 := by
  sorry

end trigonometric_equation_implications_l1509_150942


namespace arithmetic_sequence_m_value_l1509_150956

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_prop : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
    (h1 : seq.S (m - 1) = -2)
    (h2 : seq.S m = 0)
    (h3 : seq.S (m + 1) = 3) :
    m = 5 := by
  sorry

end arithmetic_sequence_m_value_l1509_150956


namespace equality_of_fractions_implies_equality_of_products_l1509_150916

theorem equality_of_fractions_implies_equality_of_products 
  (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : 
  x * (z + t + y) = z * (x + y + t) := by
  sorry

end equality_of_fractions_implies_equality_of_products_l1509_150916


namespace phenol_red_identifies_urea_decomposing_bacteria_l1509_150914

/-- Represents different types of reagents --/
inductive Reagent
  | PhenolRed
  | EMB
  | SudanIII
  | Biuret

/-- Represents a culture medium --/
structure CultureMedium where
  nitrogenSource : String
  reagent : Reagent

/-- Represents the result of a bacterial identification test --/
inductive TestResult
  | Positive
  | Negative

/-- Function to perform urea decomposition test --/
def ureaDecompositionTest (medium : CultureMedium) : TestResult := sorry

/-- Theorem stating that phenol red is the correct reagent for identifying urea-decomposing bacteria --/
theorem phenol_red_identifies_urea_decomposing_bacteria :
  ∀ (medium : CultureMedium),
    medium.nitrogenSource = "urea" →
    medium.reagent = Reagent.PhenolRed →
    ureaDecompositionTest medium = TestResult.Positive :=
  sorry

end phenol_red_identifies_urea_decomposing_bacteria_l1509_150914


namespace prism_volume_l1509_150965

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
  sorry

end prism_volume_l1509_150965


namespace problem_1_l1509_150982

theorem problem_1 : (1/3 - 3/4 + 5/6) / (1/12) = 5 := by sorry

end problem_1_l1509_150982


namespace book_pages_calculation_l1509_150986

theorem book_pages_calculation (pages_read : ℕ) (pages_unread : ℕ) (additional_pages : ℕ) :
  pages_read + pages_unread > 0 →
  pages_read = pages_unread / 3 →
  additional_pages = 48 →
  (pages_read + additional_pages : ℚ) / (pages_read + pages_unread + additional_pages) = 2/5 →
  pages_read + pages_unread = 320 := by
sorry

end book_pages_calculation_l1509_150986


namespace middle_number_proof_l1509_150950

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 15) (h4 : a + c = 20) (h5 : b + c = 23) (h6 : c = 2 * a) : 
  b = 25 / 3 := by
  sorry

end middle_number_proof_l1509_150950


namespace rational_function_value_l1509_150913

/-- A rational function with specific properties -/
structure RationalFunction where
  r : ℝ → ℝ
  s : ℝ → ℝ
  r_linear : ∃ a b : ℝ, ∀ x, r x = a * x + b
  s_quadratic : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  asymptote_neg_two : s (-2) = 0
  asymptote_three : s 3 = 0
  passes_origin : r 0 = 0 ∧ s 0 ≠ 0
  passes_one_neg_two : r 1 / s 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.r 2 / f.s 2 = -6 := by
  sorry

end rational_function_value_l1509_150913


namespace arithmetic_sequence_properties_l1509_150952

/-- An arithmetic sequence and its partial sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Partial sum sequence
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.d < 0 → ∃ M, ∀ n, seq.S n ≤ M) ∧
  ((∃ M, ∀ n, seq.S n ≤ M) → seq.d < 0) ∧
  (∃ seq : ArithmeticSequence, (∀ n, seq.S (n + 1) > seq.S n) ∧ ∃ k, seq.S k ≤ 0) ∧
  ((∀ n, seq.S n > 0) → ∀ n, seq.S (n + 1) > seq.S n) :=
by sorry

end arithmetic_sequence_properties_l1509_150952


namespace profit_decrease_calculation_l1509_150998

theorem profit_decrease_calculation (march_profit : ℝ) (april_may_decrease : ℝ) :
  march_profit > 0 →
  (march_profit * 1.3 * (1 - april_may_decrease / 100) * 1.5 = march_profit * 1.5600000000000001) →
  april_may_decrease = 20 := by
sorry

end profit_decrease_calculation_l1509_150998


namespace union_of_sets_l1509_150955

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end union_of_sets_l1509_150955


namespace distance_A_to_C_l1509_150917

/-- Prove the distance between cities A and C given travel conditions -/
theorem distance_A_to_C (time_E time_F : ℝ) (distance_AB : ℝ) (speed_ratio : ℝ) :
  time_E = 3 →
  time_F = 4 →
  distance_AB = 900 →
  speed_ratio = 4 →
  let speed_E := distance_AB / time_E
  let speed_F := speed_E / speed_ratio
  distance_AB / time_E = 4 * (distance_AB / time_E / speed_ratio) →
  speed_F * time_F = 300 :=
by sorry

end distance_A_to_C_l1509_150917


namespace flea_problem_l1509_150972

/-- Represents the number of ways a flea can reach a point given the distance and number of jumps -/
def flea_jumps (distance : ℤ) (jumps : ℕ) : ℕ := sorry

/-- Represents whether it's possible for a flea to reach a point given the distance and number of jumps -/
def flea_can_reach (distance : ℤ) (jumps : ℕ) : Prop := sorry

theorem flea_problem :
  (flea_jumps 5 7 = 7) ∧
  (flea_jumps 5 9 = 36) ∧
  ¬(flea_can_reach 2013 2028) := by sorry

end flea_problem_l1509_150972


namespace instant_noodle_change_l1509_150966

theorem instant_noodle_change (total_change : ℕ) (total_notes : ℕ) (x : ℕ) (y : ℕ) : 
  total_change = 95 →
  total_notes = 16 →
  x + y = total_notes →
  10 * x + 5 * y = total_change →
  x = 3 := by
  sorry

end instant_noodle_change_l1509_150966


namespace positive_expressions_l1509_150903

theorem positive_expressions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + x^2 * z ∧ 
  0 < y + x^2 ∧ 
  0 < y + y^2 ∧ 
  0 < y + 2 * z := by
  sorry

end positive_expressions_l1509_150903


namespace min_value_theorem_min_value_is_seven_min_value_exists_l1509_150985

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 8 / b = 2 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  2 * x + y ≥ 7 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 8 / b = 2 ∧ 2 * a + b = 7 :=
by sorry

end min_value_theorem_min_value_is_seven_min_value_exists_l1509_150985


namespace cube_split_2015_l1509_150912

/-- The number of odd numbers in the "split" of n^3, for n ≥ 2 -/
def split_count (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number, starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_2015 (m : ℕ) (hm : m > 0) :
  (∃ k, k > 0 ∧ k ≤ split_count m ∧ nth_odd k = 2015) ↔ m = 45 := by
  sorry

end cube_split_2015_l1509_150912


namespace rectangular_field_length_l1509_150987

theorem rectangular_field_length
  (area : ℝ)
  (length_increase : ℝ)
  (area_increase : ℝ)
  (h1 : area = 144)
  (h2 : length_increase = 6)
  (h3 : area_increase = 54)
  (h4 : ∀ l w, l * w = area → (l + length_increase) * w = area + area_increase) :
  ∃ l w, l * w = area ∧ l = 16 :=
sorry

end rectangular_field_length_l1509_150987


namespace sequence_inequality_l1509_150951

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, n^2 - k*n ≥ 3^2 - k*3) → 
  5 ≤ k ∧ k ≤ 7 := by
  sorry

end sequence_inequality_l1509_150951


namespace percentage_8_years_plus_is_24_percent_l1509_150970

/-- Represents the number of employees for each year range --/
structure EmployeeDistribution :=
  (less_than_2 : ℕ)
  (from_2_to_4 : ℕ)
  (from_4_to_6 : ℕ)
  (from_6_to_8 : ℕ)
  (from_8_to_10 : ℕ)
  (from_10_to_12 : ℕ)
  (from_12_to_14 : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_2 + d.from_2_to_4 + d.from_4_to_6 + d.from_6_to_8 +
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the number of employees with 8 or more years of employment --/
def employees_8_years_plus (d : EmployeeDistribution) : ℕ :=
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the percentage of employees with 8 or more years of employment --/
def percentage_8_years_plus (d : EmployeeDistribution) : ℚ :=
  (employees_8_years_plus d : ℚ) / (total_employees d : ℚ) * 100

/-- Theorem stating that the percentage of employees with 8 or more years of employment is 24% --/
theorem percentage_8_years_plus_is_24_percent (d : EmployeeDistribution)
  (h1 : d.less_than_2 = 4)
  (h2 : d.from_2_to_4 = 6)
  (h3 : d.from_4_to_6 = 5)
  (h4 : d.from_6_to_8 = 4)
  (h5 : d.from_8_to_10 = 3)
  (h6 : d.from_10_to_12 = 2)
  (h7 : d.from_12_to_14 = 1) :
  percentage_8_years_plus d = 24 := by
  sorry

end percentage_8_years_plus_is_24_percent_l1509_150970


namespace triangle_formation_with_6_and_8_l1509_150984

/-- A function that checks if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which length among given options can form a triangle with sides 6 and 8 --/
theorem triangle_formation_with_6_and_8 :
  can_form_triangle 6 8 13 ∧
  ¬(can_form_triangle 6 8 1) ∧
  ¬(can_form_triangle 6 8 2) ∧
  ¬(can_form_triangle 6 8 14) := by
  sorry


end triangle_formation_with_6_and_8_l1509_150984


namespace isosceles_base_length_l1509_150910

/-- The length of the base of an isosceles triangle, given specific conditions -/
theorem isosceles_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h1 : equilateral_perimeter = 45) 
  (h2 : isosceles_perimeter = 40) : ℝ :=
by
  -- The length of the base of the isosceles triangle is 10
  sorry

#check isosceles_base_length

end isosceles_base_length_l1509_150910


namespace two_pair_probability_l1509_150930

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Number of cards in a poker hand -/
def HandSize : ℕ := 5

/-- Number of ways to choose 5 cards from 52 -/
def TotalOutcomes : ℕ := Nat.choose StandardDeck HandSize

/-- Number of ways to form a two pair -/
def TwoPairOutcomes : ℕ := NumRanks * (Nat.choose CardsPerRank 2) * (NumRanks - 1) * (Nat.choose CardsPerRank 2) * (NumRanks - 2) * CardsPerRank

/-- Probability of forming a two pair -/
def TwoPairProbability : ℚ := TwoPairOutcomes / TotalOutcomes

theorem two_pair_probability : TwoPairProbability = 108 / 1005 := by
  sorry

end two_pair_probability_l1509_150930


namespace exists_composite_power_sum_l1509_150999

theorem exists_composite_power_sum (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, k > 1 ∧ k ∣ (x^(2^n) + y^(2^n)) :=
by sorry

end exists_composite_power_sum_l1509_150999


namespace reflected_ray_equation_l1509_150969

/-- Given an incident ray y = 2x + 1 reflected by the line y = x, 
    the equation of the reflected ray is x - 2y - 1 = 0 -/
theorem reflected_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) →  -- incident ray
  (y = x) →        -- reflecting line
  (x - 2*y - 1 = 0) -- reflected ray
  := by sorry

end reflected_ray_equation_l1509_150969


namespace power_sum_product_equals_l1509_150929

theorem power_sum_product_equals : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end power_sum_product_equals_l1509_150929


namespace xiaoying_journey_equations_l1509_150920

/-- Represents Xiaoying's journey to school --/
structure JourneyToSchool where
  totalDistance : ℝ
  totalTime : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- The system of equations representing Xiaoying's journey --/
def journeyEquations (j : JourneyToSchool) : Prop :=
  (j.uphillSpeed / 60 * j.uphillTime + j.downhillSpeed / 60 * j.downhillTime = j.totalDistance / 1000) ∧
  (j.uphillTime + j.downhillTime = j.totalTime)

/-- Theorem stating that the given conditions satisfy the journey equations --/
theorem xiaoying_journey_equations :
  ∀ (j : JourneyToSchool),
    j.totalDistance = 1200 ∧
    j.totalTime = 16 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 →
    journeyEquations j :=
by
  sorry

end xiaoying_journey_equations_l1509_150920


namespace smaller_cube_side_length_l1509_150954

/-- The side length of a smaller cube inscribed between a sphere and one face of a larger cube inscribed in the sphere. -/
theorem smaller_cube_side_length (R : ℝ) : 
  R = Real.sqrt 3 →  -- Radius of the sphere
  ∃ (x : ℝ), 
    x > 0 ∧  -- Side length of smaller cube is positive
    x < 2 ∧  -- Side length of smaller cube is less than that of larger cube
    (1 + x + x * Real.sqrt 2 / 2)^2 = 3 ∧  -- Equation derived from geometric relationships
    x = 2/3 :=
by sorry

end smaller_cube_side_length_l1509_150954


namespace ramsey_33_l1509_150915

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- A function type representing a coloring of line segments -/
def Coloring := Fin 9 → Fin 9 → Color

/-- Predicate to check if four points are coplanar -/
def are_coplanar (p₁ p₂ p₃ p₄ : Point3D) : Prop := sorry

/-- Predicate to check if a set of points forms a monochromatic triangle under a given coloring -/
def has_monochromatic_triangle (points : Fin 9 → Point3D) (coloring : Coloring) : Prop := sorry

theorem ramsey_33 (points : Fin 9 → Point3D) 
  (h_not_coplanar : ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l → 
    ¬are_coplanar (points i) (points j) (points k) (points l)) :
  ∀ coloring : Coloring, has_monochromatic_triangle points coloring := by
  sorry

end ramsey_33_l1509_150915


namespace x_zero_value_l1509_150958

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2015) → x₀ = 1 := by
  sorry

end x_zero_value_l1509_150958


namespace product_of_numbers_with_given_sum_and_difference_l1509_150926

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 3 → x * y = 154 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l1509_150926


namespace candy_problem_smallest_n_l1509_150990

theorem candy_problem (x y z n : ℕ+) : 
  (18 * x = 21 * y) ∧ (21 * y = 10 * z) ∧ (10 * z = 30 * n) → n ≥ 21 := by
  sorry

theorem smallest_n : ∃ (x y z : ℕ+), 18 * x = 21 * y ∧ 21 * y = 10 * z ∧ 10 * z = 30 * 21 := by
  sorry

end candy_problem_smallest_n_l1509_150990


namespace quadratic_factorization_l1509_150937

theorem quadratic_factorization :
  ∀ x : ℝ, 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 := by
  sorry

end quadratic_factorization_l1509_150937


namespace part_one_part_two_l1509_150964

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part (1)
theorem part_one (a x : ℝ) (h1 : a = 2) (h2 : a > 1) (h3 : p x a ∧ q x) :
  1 ≤ x ∧ x < 2 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a > 1)
  (h_necessary : ∀ x, q x → p x a)
  (h_not_sufficient : ∃ x, p x a ∧ ¬q x) :
  3 < a := by sorry

end part_one_part_two_l1509_150964


namespace polynomial_expansion_l1509_150908

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 := by
  sorry

end polynomial_expansion_l1509_150908


namespace vasya_figure_cells_l1509_150959

/-- A figure that can be cut into both 2x2 squares and zigzags of 4 cells -/
structure VasyaFigure where
  cells : ℕ
  divisible_by_4 : 4 ∣ cells
  can_cut_into_2x2 : ∃ n : ℕ, cells = 4 * n
  can_cut_into_zigzags : ∃ m : ℕ, cells = 4 * m

/-- The number of cells in Vasya's figure is a multiple of 8 and is at least 16 -/
theorem vasya_figure_cells (fig : VasyaFigure) : 
  ∃ k : ℕ, fig.cells = 8 * k ∧ fig.cells ≥ 16 := by
  sorry

end vasya_figure_cells_l1509_150959


namespace max_value_inequality_l1509_150997

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + 2*c = 2) :
  (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≤ 3/2 := by
  sorry

end max_value_inequality_l1509_150997


namespace trailing_zeros_of_product_sum_of_digits_l1509_150978

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Product of sum of digits from 1 to n -/
def product_of_sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of trailing zeros in a natural number -/
def trailing_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the product of sum of digits from 1 to 100 is 19 -/
theorem trailing_zeros_of_product_sum_of_digits : 
  trailing_zeros (product_of_sum_of_digits 100) = 19 := by sorry

end trailing_zeros_of_product_sum_of_digits_l1509_150978


namespace intersection_N_complement_M_l1509_150979

-- Define the universe U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_N_complement_M : N ∩ (U \ M) = {3, 5} := by
  sorry

end intersection_N_complement_M_l1509_150979


namespace four_numbers_sum_l1509_150994

theorem four_numbers_sum (a b c d T : ℝ) (h : a + b + c + d = T) :
  3 * ((a + 1) + (b + 1) + (c + 1) + (d + 1)) = 3 * T + 12 := by
  sorry

end four_numbers_sum_l1509_150994


namespace dinner_time_calculation_l1509_150924

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  let newHour := (totalMinutes / 60) % 24
  let newMinute := totalMinutes % 60
  ⟨newHour, newMinute, by sorry, by sorry⟩

theorem dinner_time_calculation (start : Time) 
    (h_start : start = ⟨16, 0, by sorry, by sorry⟩)
    (commute : Nat) (h_commute : commute = 30)
    (grocery : Nat) (h_grocery : grocery = 30)
    (drycleaning : Nat) (h_drycleaning : drycleaning = 10)
    (dog : Nat) (h_dog : dog = 20)
    (cooking : Nat) (h_cooking : cooking = 90) :
  addMinutes start (commute + grocery + drycleaning + dog + cooking) = ⟨19, 0, by sorry, by sorry⟩ := by
  sorry

end dinner_time_calculation_l1509_150924


namespace rogers_candy_problem_l1509_150927

/-- Roger's candy problem -/
theorem rogers_candy_problem (initial_candies given_candies remaining_candies : ℕ) :
  given_candies = 3 →
  remaining_candies = 92 →
  initial_candies = remaining_candies + given_candies →
  initial_candies = 95 :=
by sorry

end rogers_candy_problem_l1509_150927


namespace parallel_lines_a_value_l1509_150949

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The problem statement -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel (1 + a) 1 1 2 a 2 → a = 1 ∨ a = -2 := by
  sorry

end parallel_lines_a_value_l1509_150949


namespace arithmetic_sequence_proof_l1509_150968

theorem arithmetic_sequence_proof :
  ∀ (a : ℕ → ℤ),
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence condition
    (a 0 = 3^2) →  -- first term is 3²
    (a 2 = 3^4) →  -- third term is 3⁴
    (a 1 = 33 ∧ a 3 = 105) :=
by
  sorry

end arithmetic_sequence_proof_l1509_150968


namespace count_special_numbers_eq_384_l1509_150928

/-- A function that counts the number of 4-digit numbers beginning with 2 
    and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let non_two_digits := digits \ {2}
  let count_with_two_twos := 3 * (Finset.card non_two_digits - 1) * (Finset.card non_two_digits - 1)
  let count_with_non_two_pairs := 3 * (Finset.card non_two_digits) * (Finset.card non_two_digits - 1)
  count_with_two_twos + count_with_non_two_pairs

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end count_special_numbers_eq_384_l1509_150928


namespace total_broken_bulbs_to_replace_l1509_150967

/-- Represents the number of broken light bulbs that need to be replaced -/
def broken_bulbs_to_replace (kitchen_bulbs foyer_broken_bulbs living_room_bulbs : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_bulbs) / 5
  let foyer_broken := foyer_broken_bulbs
  let living_room_broken := living_room_bulbs / 2
  kitchen_broken + foyer_broken + living_room_broken

/-- Theorem stating the total number of broken light bulbs to be replaced -/
theorem total_broken_bulbs_to_replace :
  broken_bulbs_to_replace 35 10 24 = 43 :=
by
  sorry

end total_broken_bulbs_to_replace_l1509_150967


namespace karen_grooms_one_chihuahua_l1509_150918

/-- The time it takes to groom a Rottweiler -/
def rottweiler_time : ℕ := 20

/-- The time it takes to groom a border collie -/
def border_collie_time : ℕ := 10

/-- The time it takes to groom a chihuahua -/
def chihuahua_time : ℕ := 45

/-- The total time Karen spends grooming -/
def total_time : ℕ := 255

/-- The number of Rottweilers Karen grooms -/
def num_rottweilers : ℕ := 6

/-- The number of border collies Karen grooms -/
def num_border_collies : ℕ := 9

/-- The number of chihuahuas Karen grooms -/
def num_chihuahuas : ℕ := 1

theorem karen_grooms_one_chihuahua :
  num_chihuahuas * chihuahua_time =
  total_time - (num_rottweilers * rottweiler_time + num_border_collies * border_collie_time) :=
by sorry

end karen_grooms_one_chihuahua_l1509_150918


namespace employed_males_percentage_l1509_150933

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 75)
  : (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100 = 15) := by
  sorry

end employed_males_percentage_l1509_150933


namespace grandson_age_l1509_150975

theorem grandson_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  (grandson_age + 4) + (grandfather_age + 4) = 78 →
  grandson_age = 10 := by
  sorry

end grandson_age_l1509_150975


namespace intersection_x_coordinate_l1509_150935

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 1
def line2 (x y : ℝ) : Prop := 5 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 99 / 8 := by
  sorry

end intersection_x_coordinate_l1509_150935


namespace find_a20_l1509_150922

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem find_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 20 = -30 := by
  sorry

end find_a20_l1509_150922


namespace michael_truck_meetings_l1509_150901

/-- Represents the number of meetings between Michael and the garbage truck --/
def number_of_meetings : ℕ := 7

/-- Michael's walking speed in feet per second --/
def michael_speed : ℝ := 6

/-- Distance between trash pails in feet --/
def pail_distance : ℝ := 200

/-- Garbage truck's speed in feet per second --/
def truck_speed : ℝ := 10

/-- Time the truck stops at each pail in seconds --/
def truck_stop_time : ℝ := 40

/-- Initial distance between Michael and the truck in feet --/
def initial_distance : ℝ := 250

/-- Theorem stating that Michael and the truck will meet 7 times --/
theorem michael_truck_meetings :
  ∃ (t : ℝ), t > 0 ∧
  (michael_speed * t = truck_speed * (t - truck_stop_time * (number_of_meetings - 1)) + initial_distance) :=
sorry

end michael_truck_meetings_l1509_150901


namespace special_haircut_price_l1509_150974

/-- Represents the cost of different types of haircuts and the hairstylist's earnings --/
structure HaircutPrices where
  normal : ℝ
  special : ℝ
  trendy : ℝ
  daily_normal : ℕ
  daily_special : ℕ
  daily_trendy : ℕ
  weekly_earnings : ℝ
  days_per_week : ℕ

/-- Theorem stating that the special haircut price is $6 given the conditions --/
theorem special_haircut_price (h : HaircutPrices) 
    (h_normal : h.normal = 5)
    (h_trendy : h.trendy = 8)
    (h_daily_normal : h.daily_normal = 5)
    (h_daily_special : h.daily_special = 3)
    (h_daily_trendy : h.daily_trendy = 2)
    (h_weekly_earnings : h.weekly_earnings = 413)
    (h_days_per_week : h.days_per_week = 7) :
  h.special = 6 := by
  sorry

#check special_haircut_price

end special_haircut_price_l1509_150974


namespace max_value_of_f_l1509_150923

open Real

noncomputable def f (x : ℝ) := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = 1 / Real.exp 1 :=
sorry

end max_value_of_f_l1509_150923


namespace magic_wheel_product_l1509_150996

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_two_odd_between (a d : ℕ) : Prop :=
  ∃ b c : ℕ, a < b ∧ b < c ∧ c < d ∧
  ¬(is_even b) ∧ ¬(is_even c) ∧
  (d - a) % 16 = 3

theorem magic_wheel_product :
  ∀ a d : ℕ,
  1 ≤ a ∧ a ≤ 16 ∧
  1 ≤ d ∧ d ≤ 16 ∧
  is_even a ∧
  is_even d ∧
  has_two_odd_between a d →
  a * d = 120 :=
sorry

end magic_wheel_product_l1509_150996


namespace no_common_root_l1509_150961

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, x₀^2 + b*x₀ + c = 0 ∧ x₀^2 + a*x₀ + d = 0 := by
  sorry

end no_common_root_l1509_150961


namespace subtraction_reciprocal_l1509_150938

theorem subtraction_reciprocal (x y : ℝ) (h : x - y = 3 * x * y) :
  1 / x - 1 / y = -3 :=
by sorry

end subtraction_reciprocal_l1509_150938


namespace inverse_34_mod_47_l1509_150925

theorem inverse_34_mod_47 (h : (13⁻¹ : ZMod 47) = 29) : (34⁻¹ : ZMod 47) = 18 := by
  sorry

end inverse_34_mod_47_l1509_150925


namespace sum_of_simplified_fraction_75_135_l1509_150904

def simplify_fraction (n d : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd n d
  (n / g, d / g)

theorem sum_of_simplified_fraction_75_135 :
  let (n, d) := simplify_fraction 75 135
  n + d = 14 := by sorry

end sum_of_simplified_fraction_75_135_l1509_150904


namespace fraction_decimal_places_l1509_150947

/-- The number of decimal places when converting the fraction 123456789 / (2^26 * 5^4) to a decimal -/
def decimal_places : ℕ :=
  let numerator : ℕ := 123456789
  let denominator : ℕ := 2^26 * 5^4
  26

theorem fraction_decimal_places :
  decimal_places = 26 :=
sorry

end fraction_decimal_places_l1509_150947


namespace triangle_inequality_l1509_150921

/-- Given a triangle with side lengths a, b, and c, 
    prove the inequality and its equality condition --/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0) ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) := by
  sorry

end triangle_inequality_l1509_150921


namespace amanda_candy_bars_l1509_150983

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_day_given : ℕ := 3
  let second_day_given : ℕ := 4 * first_day_given
  let kept_for_self : ℕ := 22
  let bought_next_day : ℕ := kept_for_self + second_day_given - (initial_bars - first_day_given)
  bought_next_day = 30 := by sorry

end amanda_candy_bars_l1509_150983


namespace percentage_green_shirts_l1509_150907

/-- The percentage of students wearing green shirts in a school, given the following conditions:
  * The total number of students is 700
  * 45% of students wear blue shirts
  * 23% of students wear red shirts
  * 119 students wear colors other than blue, red, or green
-/
theorem percentage_green_shirts (total : ℕ) (blue_percent red_percent : ℚ) (other : ℕ) :
  total = 700 →
  blue_percent = 45 / 100 →
  red_percent = 23 / 100 →
  other = 119 →
  (((total : ℚ) - (blue_percent * total + red_percent * total + other)) / total) * 100 = 15 := by
  sorry

end percentage_green_shirts_l1509_150907


namespace no_integer_solutions_l1509_150953

theorem no_integer_solutions : ¬∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2011 := by
  sorry

end no_integer_solutions_l1509_150953


namespace joe_journey_time_l1509_150905

/-- Represents Joe's journey from home to the store -/
structure JoeJourney where
  walk_speed : ℝ
  run_speed : ℝ
  walk_time : ℝ
  total_distance : ℝ

/-- Theorem: Joe's total journey time is 15 minutes -/
theorem joe_journey_time (j : JoeJourney) 
  (h1 : j.run_speed = 2 * j.walk_speed)
  (h2 : j.walk_time = 10)
  (h3 : j.total_distance = 2 * (j.walk_speed * j.walk_time)) : 
  j.walk_time + (j.total_distance / 2) / j.run_speed = 15 := by
  sorry

#check joe_journey_time

end joe_journey_time_l1509_150905


namespace rhombus_area_in_square_l1509_150900

/-- The area of a rhombus formed by intersecting equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height : ℝ := square_side * (Real.sqrt 3) / 2
  let rhombus_diagonal1 : ℝ := 2 * triangle_height - square_side
  let rhombus_diagonal2 : ℝ := square_side
  let rhombus_area : ℝ := (rhombus_diagonal1 * rhombus_diagonal2) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 := by
  sorry


end rhombus_area_in_square_l1509_150900


namespace added_amount_l1509_150995

theorem added_amount (x : ℝ) (y : ℝ) (h1 : x = 6) (h2 : 2 / 3 * x + y = 10) : y = 6 := by
  sorry

end added_amount_l1509_150995


namespace college_selection_ways_l1509_150902

theorem college_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 2 →
  (m * (n - m).choose (k - 1)) + ((n - m).choose k) = 16 := by sorry

end college_selection_ways_l1509_150902


namespace toms_living_room_length_l1509_150946

def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250
def boxes_needed : ℕ := 7

theorem toms_living_room_length : 
  (flooring_laid + boxes_needed * flooring_per_box) / room_width = 16 := by
  sorry

end toms_living_room_length_l1509_150946


namespace triangle_is_equilateral_l1509_150931

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c

def condition2 (t : Triangle) : Prop :=
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.A = Real.pi / 3

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : is_equilateral t := by
  sorry

end triangle_is_equilateral_l1509_150931


namespace solution_count_l1509_150988

/-- The number of positive integer solutions for a system of equations involving a prime number -/
def num_solutions (p : ℕ) : ℕ :=
  if p = 2 then 5
  else if p % 4 = 1 then 11
  else 3

/-- The main theorem stating the number of solutions for the given system of equations -/
theorem solution_count (p : ℕ) (hp : Nat.Prime p) :
  (∃ (n : ℕ), n = (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a * c + b * d = p * (a + c) ∧
    b * c - a * d = p * (b - d))
    (Finset.product (Finset.range (p^3 + 1)) (Finset.product (Finset.range (p^3 + 1))
      (Finset.product (Finset.range (p^3 + 1)) (Finset.range (p^3 + 1)))))).card) ∧
  n = num_solutions p :=
sorry

end solution_count_l1509_150988


namespace cats_remaining_l1509_150960

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end cats_remaining_l1509_150960


namespace cuboid_diagonal_l1509_150909

/-- Given a cuboid with dimensions a, b, and c, if its surface area is 11
    and the sum of the lengths of its twelve edges is 24,
    then the length of its diagonal is 5. -/
theorem cuboid_diagonal (a b c : ℝ) 
    (h1 : 2 * (a * b + b * c + a * c) = 11)  -- surface area condition
    (h2 : 4 * (a + b + c) = 24) :            -- sum of edges condition
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end cuboid_diagonal_l1509_150909


namespace positive_real_inequalities_l1509_150957

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end positive_real_inequalities_l1509_150957


namespace exists_n_congruence_l1509_150973

theorem exists_n_congruence (l : ℕ+) : ∃ n : ℕ, (n^n + 47) % (2^l.val) = 0 := by
  sorry

end exists_n_congruence_l1509_150973


namespace melanie_dimes_l1509_150940

theorem melanie_dimes (initial : Nat) (dad_gift : Nat) (final_total : Nat) 
  (h1 : initial = 19)
  (h2 : dad_gift = 39)
  (h3 : final_total = 83) :
  final_total - (initial + dad_gift) = 25 := by
  sorry

end melanie_dimes_l1509_150940


namespace vector_properties_l1509_150981

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- Define the conditions
def non_collinear (a b : V) : Prop := ¬ ∃ (k : ℝ), a = k • b
def same_starting_point (a b : V) : Prop := True  -- This is implicitly assumed in the vector space
def equal_magnitude (a b : V) : Prop := ‖a‖ = ‖b‖
def angle_60_degrees (a b : V) : Prop := inner a b = (1/2 : ℝ) * ‖a‖ * ‖b‖

-- Define the theorem
theorem vector_properties
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : non_collinear V a b)
  (h4 : same_starting_point V a b)
  (h5 : equal_magnitude V a b)
  (h6 : angle_60_degrees V a b) :
  (∃ (k : ℝ), k • ((1/2 : ℝ) • b - a) = (1/3 : ℝ) • b - (2/3 : ℝ) • a) ∧
  (∀ (t : ℝ), ‖a - (1/2 : ℝ) • b‖ ≤ ‖a - t • b‖) :=
sorry

end vector_properties_l1509_150981
