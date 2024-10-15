import Mathlib

namespace NUMINAMATH_CALUDE_principal_calculation_l3471_347192

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal is 10040.625 -/
theorem principal_calculation :
  let interest : ℚ := 4016.25
  let rate : ℚ := 8
  let time : ℚ := 5
  calculate_principal interest rate time = 10040.625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3471_347192


namespace NUMINAMATH_CALUDE_inconsistent_fraction_problem_l3471_347148

theorem inconsistent_fraction_problem :
  ¬ ∃ (f : ℚ), (f * 4 = 8) ∧ ((1/8) * 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_fraction_problem_l3471_347148


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3471_347174

theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 (π / 2) →
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3471_347174


namespace NUMINAMATH_CALUDE_mixture_volume_l3471_347125

theorem mixture_volume (V : ℝ) (h1 : V > 0) : 
  (0.20 * V = 0.15 * (V + 5)) → V = 15 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l3471_347125


namespace NUMINAMATH_CALUDE_no_real_solutions_l3471_347121

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3471_347121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3471_347129

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum : 
  2 * arithmetic_sum 51 99 2 = 3750 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3471_347129


namespace NUMINAMATH_CALUDE_two_digit_sum_product_l3471_347176

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: If c is a 2-digit positive integer where the sum of its digits is 10
    and the product of its digits is 25, then c = 55 -/
theorem two_digit_sum_product (c : ℕ) : 
  10 ≤ c ∧ c ≤ 99 ∧ 
  tens_digit c + ones_digit c = 10 ∧
  tens_digit c * ones_digit c = 25 →
  c = 55 := by
  sorry


end NUMINAMATH_CALUDE_two_digit_sum_product_l3471_347176


namespace NUMINAMATH_CALUDE_iris_rose_ratio_l3471_347107

theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) : 
  initial_roses = 42 →
  added_roses = 35 →
  (3 : ℚ) / 7 = (irises_needed : ℚ) / (initial_roses + added_roses) →
  irises_needed = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_iris_rose_ratio_l3471_347107


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3471_347149

theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 80)
  (h2 : added_water = 28)
  (h3 : final_fraction = 3/4)
  (h4 : final_fraction * capacity = (initial_percentage / 100) * capacity + added_water) :
  initial_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l3471_347149


namespace NUMINAMATH_CALUDE_seashell_theorem_l3471_347147

def seashell_problem (mary_shells jessica_shells : ℕ) : Prop :=
  let kevin_shells := 3 * mary_shells
  let laura_shells := jessica_shells / 2
  mary_shells + jessica_shells + kevin_shells + laura_shells = 134

theorem seashell_theorem :
  seashell_problem 18 41 := by sorry

end NUMINAMATH_CALUDE_seashell_theorem_l3471_347147


namespace NUMINAMATH_CALUDE_certain_number_equation_l3471_347172

theorem certain_number_equation (x : ℝ) : (10 + 20 + 60) / 3 = ((10 + x + 25) / 3) + 5 ↔ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3471_347172


namespace NUMINAMATH_CALUDE_writing_outlining_difference_l3471_347135

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ

/-- Defines the conditions for Javier's speech preparation --/
def javierSpeechConditions (t : SpeechTime) : Prop :=
  t.outlining = 30 ∧
  t.writing > t.outlining ∧
  t.practicing = t.writing / 2 ∧
  t.outlining + t.writing + t.practicing = 117

/-- Theorem stating the difference between writing and outlining time --/
theorem writing_outlining_difference (t : SpeechTime) 
  (h : javierSpeechConditions t) : t.writing - t.outlining = 28 := by
  sorry

#check writing_outlining_difference

end NUMINAMATH_CALUDE_writing_outlining_difference_l3471_347135


namespace NUMINAMATH_CALUDE_rocks_needed_l3471_347167

/-- The number of rocks Mrs. Hilt needs for her garden border -/
def total_rocks_needed : ℕ := 125

/-- The number of rocks Mrs. Hilt currently has -/
def rocks_on_hand : ℕ := 64

/-- Theorem: Mrs. Hilt needs 61 more rocks to complete her garden border -/
theorem rocks_needed : total_rocks_needed - rocks_on_hand = 61 := by
  sorry

end NUMINAMATH_CALUDE_rocks_needed_l3471_347167


namespace NUMINAMATH_CALUDE_addison_sunday_ticket_sales_l3471_347154

/-- Proves that Addison sold 78 raffle tickets on Sunday given the conditions of the problem -/
theorem addison_sunday_ticket_sales : 
  ∀ (friday saturday sunday : ℕ),
  friday = 181 →
  saturday = 2 * friday →
  saturday = sunday + 284 →
  sunday = 78 := by sorry

end NUMINAMATH_CALUDE_addison_sunday_ticket_sales_l3471_347154


namespace NUMINAMATH_CALUDE_max_intersection_points_l3471_347123

/-- Represents a line segment -/
structure Segment where
  id : ℕ

/-- Represents an intersection point -/
structure IntersectionPoint where
  id : ℕ

/-- The set of all segments -/
def segments : Finset Segment :=
  sorry

/-- The set of all intersection points -/
def intersectionPoints : Finset IntersectionPoint :=
  sorry

/-- Function that returns the number of intersections for a given segment -/
def intersectionsForSegment (s : Segment) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (segments.card = 10) →
  (∀ s ∈ segments, intersectionsForSegment s = 3) →
  intersectionPoints.card ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l3471_347123


namespace NUMINAMATH_CALUDE_table_relationship_l3471_347127

def f (x : ℝ) : ℝ := 21 - x^2

theorem table_relationship : 
  (f 0 = 21) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 16) ∧ 
  (f 3 = 9) ∧ 
  (f 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_table_relationship_l3471_347127


namespace NUMINAMATH_CALUDE_problem_statement_l3471_347181

theorem problem_statement (x y z : ℚ) (hx : x = 4/3) (hy : y = 3/4) (hz : z = 3/2) :
  (1/2) * x^6 * y^7 * z^4 = 243/128 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3471_347181


namespace NUMINAMATH_CALUDE_prism_21_edges_l3471_347139

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := sorry

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := sorry

/-- Theorem: A prism with 21 edges has 9 faces and 7 vertices -/
theorem prism_21_edges (p : Prism) (h : p.edges = 21) : 
  num_faces p = 9 ∧ num_vertices p = 7 := by sorry

end NUMINAMATH_CALUDE_prism_21_edges_l3471_347139


namespace NUMINAMATH_CALUDE_east_west_southwest_angle_l3471_347168

/-- Represents the directions of rays in the decagon arrangement -/
inductive Direction
| North
| East
| WestSouthWest

/-- Represents a regular decagon with rays -/
structure DecagonArrangement where
  rays : Fin 10 → Direction
  north_ray : ∃ i, rays i = Direction.North

/-- Calculates the number of sectors between two directions -/
def sectors_between (d1 d2 : Direction) : ℕ := sorry

/-- Calculates the angle in degrees between two rays -/
def angle_between (d1 d2 : Direction) : ℝ :=
  (sectors_between d1 d2 : ℝ) * 36

theorem east_west_southwest_angle (arrangement : DecagonArrangement) :
  angle_between Direction.East Direction.WestSouthWest = 180 := by sorry

end NUMINAMATH_CALUDE_east_west_southwest_angle_l3471_347168


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l3471_347194

def temperature_problem (t1 t2 t3 t4 : ℤ) : Prop :=
  let temps := [t1, t2, t3, t4]
  (t1 = -36) ∧ (t2 = 13) ∧ (t3 = -10) ∧ 
  (temps.sum / temps.length = -12) ∧
  (t4 = -15)

theorem fourth_day_temperature :
  ∃ t4 : ℤ, temperature_problem (-36) 13 (-10) t4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l3471_347194


namespace NUMINAMATH_CALUDE_sakshi_tanya_efficiency_increase_l3471_347103

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Theorem stating the efficiency increase from Sakshi to Tanya -/
theorem sakshi_tanya_efficiency_increase :
  let sakshi_rate : ℚ := 1/5
  let tanya_rate : ℚ := 1/4
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_sakshi_tanya_efficiency_increase_l3471_347103


namespace NUMINAMATH_CALUDE_problem_solution_l3471_347133

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3471_347133


namespace NUMINAMATH_CALUDE_two_sets_satisfying_union_condition_l3471_347146

theorem two_sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    S.card = 2 ∧ 
    ∀ M ∈ S, M ∪ {1} = {1, 2, 3} ∧
    ∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S :=
by sorry

end NUMINAMATH_CALUDE_two_sets_satisfying_union_condition_l3471_347146


namespace NUMINAMATH_CALUDE_inequality_proof_l3471_347110

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3471_347110


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3471_347126

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 3)
  let b : ℝ → ℝ × ℝ := λ x ↦ (6, x)
  ∀ x : ℝ, perpendicular a (b x) → x = -8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3471_347126


namespace NUMINAMATH_CALUDE_body_part_count_l3471_347124

theorem body_part_count (suspension_days_per_instance : ℕ) 
                        (total_bullying_instances : ℕ) 
                        (body_part_count : ℕ) : 
  suspension_days_per_instance = 3 →
  total_bullying_instances = 20 →
  suspension_days_per_instance * total_bullying_instances = 3 * body_part_count →
  body_part_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_body_part_count_l3471_347124


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l3471_347185

def initial_amount : ℚ := 90

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (sandwich_fraction * initial_amount + museum_fraction * initial_amount + book_fraction * initial_amount)

theorem jennifer_remaining_money :
  remaining_amount = 12 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l3471_347185


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l3471_347144

/-- Given a function f(x) = ax^2 + bx - ln(x) where a > 0 and b ∈ ℝ,
    if f(x) ≥ f(1) for all x > 0, then ln(a) < -2b -/
theorem function_minimum_implies_inequality 
  (a b : ℝ) 
  (ha : a > 0)
  (hf : ∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) :
  Real.log a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l3471_347144


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3471_347151

theorem complex_number_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.mk (Real.sin 3) (Real.cos 3) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3471_347151


namespace NUMINAMATH_CALUDE_clowns_in_mobiles_l3471_347162

/-- Given a number of clown mobiles and a total number of clowns,
    calculate the number of clowns in each mobile assuming even distribution -/
def clowns_per_mobile (num_mobiles : ℕ) (total_clowns : ℕ) : ℕ :=
  total_clowns / num_mobiles

/-- Theorem stating that with 5 clown mobiles and 140 clowns in total,
    there are 28 clowns in each mobile -/
theorem clowns_in_mobiles :
  clowns_per_mobile 5 140 = 28 := by
  sorry


end NUMINAMATH_CALUDE_clowns_in_mobiles_l3471_347162


namespace NUMINAMATH_CALUDE_circle_max_min_linear_function_l3471_347166

theorem circle_max_min_linear_function :
  ∀ x y : ℝ, x^2 + y^2 = 16*x + 8*y + 20 →
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≤ 116) ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≥ -64) ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = 16*x₁ + 8*y₁ + 20 ∧ 4*x₁ + 3*y₁ = 116) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + y₂^2 = 16*x₂ + 8*y₂ + 20 ∧ 4*x₂ + 3*y₂ = -64) :=
by sorry


end NUMINAMATH_CALUDE_circle_max_min_linear_function_l3471_347166


namespace NUMINAMATH_CALUDE_correct_operation_l3471_347128

theorem correct_operation (a : ℝ) : 2 * a + 3 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3471_347128


namespace NUMINAMATH_CALUDE_distance_point_to_parametric_line_l3471_347173

/-- The distance from a point to a line defined parametrically -/
theorem distance_point_to_parametric_line :
  let P : ℝ × ℝ := (2, 0)
  let line (t : ℝ) : ℝ × ℝ := (1 + 4*t, 2 + 3*t)
  let distance (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : ℝ :=
    -- Define distance function here (implementation not provided)
    sorry
  distance P line = 11/5 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_parametric_line_l3471_347173


namespace NUMINAMATH_CALUDE_min_sum_three_digit_numbers_l3471_347169

def is_valid_triple (a b c : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧ 
  c ≥ 100 ∧ c < 1000 ∧ 
  a + b = c

def uses_distinct_digits (a b c : Nat) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10
  digits.length = 9 ∧ digits.toFinset.card = 9 ∧ 
  ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 9

theorem min_sum_three_digit_numbers :
  ∃ a b c : Nat, is_valid_triple a b c ∧ 
  uses_distinct_digits a b c ∧
  (∀ x y z : Nat, is_valid_triple x y z → uses_distinct_digits x y z → 
    a + b + c ≤ x + y + z) ∧
  a + b + c = 459 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_three_digit_numbers_l3471_347169


namespace NUMINAMATH_CALUDE_volleyball_tournament_wins_l3471_347137

theorem volleyball_tournament_wins (n m : ℕ) : 
  ∀ (x : ℕ), 0 < x ∧ x < 73 →
  x * n + (73 - x) * m = 36 * 73 →
  n = m := by
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_wins_l3471_347137


namespace NUMINAMATH_CALUDE_max_visible_sum_l3471_347134

def cube_numbers : List ℕ := [1, 3, 9, 27, 81, 243]

def is_valid_cube (c : List ℕ) : Prop :=
  c.length = 6 ∧ c.toFinset = cube_numbers.toFinset

def visible_sum (bottom middle top : List ℕ) : ℕ :=
  (bottom.take 5).sum + (middle.take 5).sum + (top.take 5).sum

def is_valid_stack (bottom middle top : List ℕ) : Prop :=
  is_valid_cube bottom ∧ is_valid_cube middle ∧ is_valid_cube top

theorem max_visible_sum :
  ∀ bottom middle top : List ℕ,
    is_valid_stack bottom middle top →
    visible_sum bottom middle top ≤ 1087 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l3471_347134


namespace NUMINAMATH_CALUDE_impossible_grouping_l3471_347106

theorem impossible_grouping : ¬ ∃ (partition : List (List Nat)),
  (∀ group ∈ partition, (∀ n ∈ group, 1 ≤ n ∧ n ≤ 77)) ∧
  (∀ group ∈ partition, group.length ≥ 3) ∧
  (∀ group ∈ partition, ∃ n ∈ group, n = (group.sum - n)) ∧
  (partition.join.toFinset = Finset.range 77) :=
by sorry

end NUMINAMATH_CALUDE_impossible_grouping_l3471_347106


namespace NUMINAMATH_CALUDE_function_range_implies_m_range_l3471_347132

-- Define the function f(x) = x^2 - 2x + 5
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the theorem
theorem function_range_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_implies_m_range_l3471_347132


namespace NUMINAMATH_CALUDE_x_value_proof_l3471_347196

theorem x_value_proof : 
  ∀ x : ℝ, x = 88 * (1 + 25 / 100) → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3471_347196


namespace NUMINAMATH_CALUDE_antons_winning_strategy_l3471_347195

theorem antons_winning_strategy :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
  (∀ x : ℕ, 
    let n := f x
    ¬ ∃ m : ℕ, n = m * m ∧  -- n is not a perfect square
    ∃ k : ℕ, n + (n + 1) + (n + 2) = k * k) -- sum of three consecutive numbers starting from n is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_antons_winning_strategy_l3471_347195


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3471_347117

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2*x - 6 + Real.log x

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
by sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3471_347117


namespace NUMINAMATH_CALUDE_solve_equation_l3471_347190

theorem solve_equation : ∃ x : ℝ, 0.035 * x = 42 ∧ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3471_347190


namespace NUMINAMATH_CALUDE_infiniteLoopDecimal_eq_fraction_l3471_347152

/-- Represents the infinite loop decimal 0.0 ̇1 ̇7 -/
def infiniteLoopDecimal : ℚ := sorry

/-- The infinite loop decimal 0.0 ̇1 ̇7 is equal to 17/990 -/
theorem infiniteLoopDecimal_eq_fraction : infiniteLoopDecimal = 17 / 990 := by sorry

end NUMINAMATH_CALUDE_infiniteLoopDecimal_eq_fraction_l3471_347152


namespace NUMINAMATH_CALUDE_mike_work_hours_l3471_347156

/-- Given that Mike worked 3 hours each day for 5 days, prove that his total work hours is 15. -/
theorem mike_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → total_hours = hours_per_day * days_worked → total_hours = 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l3471_347156


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_solution_a_l3471_347159

/-- Proves that the percentage of alcohol in Solution A is 27% given the specified conditions. -/
theorem alcohol_percentage_in_solution_a : ∀ x : ℝ,
  -- Solution A has 6 liters of water and x% of alcohol
  -- Solution B has 9 liters of a solution containing 57% alcohol
  -- After mixing, the new mixture has 45% alcohol concentration
  (6 * x + 9 * 0.57 = 15 * 0.45) →
  x = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_solution_a_l3471_347159


namespace NUMINAMATH_CALUDE_sum_of_squares_not_divisible_by_17_l3471_347113

theorem sum_of_squares_not_divisible_by_17 (x y z : ℤ) :
  Nat.Coprime x.natAbs y.natAbs ∧
  Nat.Coprime x.natAbs z.natAbs ∧
  Nat.Coprime y.natAbs z.natAbs →
  (x + y + z) % 17 = 0 →
  (x * y * z) % 17 = 0 →
  (x^2 + y^2 + z^2) % 17 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_divisible_by_17_l3471_347113


namespace NUMINAMATH_CALUDE_uniform_purchase_theorem_l3471_347175

/-- Represents the price per set based on the number of sets purchased -/
def price_per_set (n : ℕ) : ℕ :=
  if n ≤ 50 then 50
  else if n ≤ 90 then 40
  else 30

/-- The total number of students in both classes -/
def total_students : ℕ := 92

/-- The range of students in Class A -/
def class_a_range (n : ℕ) : Prop := 51 < n ∧ n < 55

/-- The total amount paid when classes purchase uniforms separately -/
def separate_purchase_total : ℕ := 4080

/-- Theorem stating the number of students in each class and the most cost-effective plan -/
theorem uniform_purchase_theorem (class_a class_b : ℕ) :
  class_a + class_b = total_students →
  class_a_range class_a →
  price_per_set class_a * class_a + price_per_set class_b * class_b = separate_purchase_total →
  (class_a = 52 ∧ class_b = 40) ∧
  price_per_set 91 * 91 = 2730 ∧
  ∀ n : ℕ, n ≤ total_students - 8 → price_per_set 91 * 91 ≤ price_per_set n * n :=
by sorry

end NUMINAMATH_CALUDE_uniform_purchase_theorem_l3471_347175


namespace NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l3471_347193

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] :=
  (A B : α)

/-- The set of points P such that PA + PB = 2AB -/
def EllipseSet (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) : Set α :=
  {P : α | ‖P - points.A‖ + ‖P - points.B‖ = 2 * ‖points.A - points.B‖}

/-- Definition of an ellipse with given foci and major axis -/
def Ellipse (α : Type*) [NormedAddCommGroup α] (F₁ F₂ : α) (major_axis : ℝ) : Set α :=
  {P : α | ‖P - F₁‖ + ‖P - F₂‖ = major_axis}

/-- Theorem stating that the set of points P such that PA + PB = 2AB 
    forms an ellipse with A and B as foci and major axis 2AB -/
theorem ellipse_set_is_ellipse (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) :
  EllipseSet α points = Ellipse α points.A points.B (2 * ‖points.A - points.B‖) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l3471_347193


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3471_347182

theorem trigonometric_expression_value (α : ℝ) 
  (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) :
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3 / 40 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3471_347182


namespace NUMINAMATH_CALUDE_sad_children_count_l3471_347112

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) :
  total = 60 →
  happy = 30 →
  neither = 20 →
  boys = 22 →
  girls = 38 →
  happy_boys = 6 →
  sad_girls = 4 →
  neither_boys = 10 →
  total = happy + neither + (total - happy - neither) →
  total - happy - neither = 10 := by
sorry

end NUMINAMATH_CALUDE_sad_children_count_l3471_347112


namespace NUMINAMATH_CALUDE_xyz_sum_eq_32_l3471_347188

theorem xyz_sum_eq_32 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 64) :
  x*y + y*z + x*z = 32 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_eq_32_l3471_347188


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l3471_347165

/-- The area of the region bounded by x = 2, y = 2, and the coordinate axes is 4 square units. -/
theorem area_of_bounded_region : 
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  ∃ (A : Set (ℝ × ℝ)), A = region ∧ MeasureTheory.volume A = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l3471_347165


namespace NUMINAMATH_CALUDE_B_parity_2021_2022_2023_l3471_347122

def B : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => B (n + 2) + B (n + 1)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem B_parity_2021_2022_2023 :
  is_odd (B 2021) ∧ is_odd (B 2022) ∧ ¬is_odd (B 2023) := by sorry

end NUMINAMATH_CALUDE_B_parity_2021_2022_2023_l3471_347122


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3471_347116

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-5)^2 + (y+3)^2) = 10

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (0, 1)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (5, -3)

/-- The constant sum of distances from any point on the ellipse to the foci -/
def constant_sum : ℝ := 10

/-- Theorem stating that the given equation describes an ellipse -/
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3471_347116


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3471_347100

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 3*x = 4) ↔ (∀ x : ℝ, x^2 + 3*x ≠ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3471_347100


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l3471_347170

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples / 12) * 3

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 :=
by sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l3471_347170


namespace NUMINAMATH_CALUDE_solid_is_triangular_prism_l3471_347157

/-- Represents a three-dimensional solid -/
structure Solid :=
  (front_view : Shape)
  (side_view : Shape)

/-- Represents geometric shapes -/
inductive Shape
  | Triangle
  | Quadrilateral
  | Other

/-- Defines a triangular prism -/
def is_triangular_prism (s : Solid) : Prop :=
  s.front_view = Shape.Triangle ∧ s.side_view = Shape.Quadrilateral

/-- Theorem: A solid with triangular front view and quadrilateral side view is a triangular prism -/
theorem solid_is_triangular_prism (s : Solid) 
  (h1 : s.front_view = Shape.Triangle) 
  (h2 : s.side_view = Shape.Quadrilateral) : 
  is_triangular_prism s := by
  sorry

end NUMINAMATH_CALUDE_solid_is_triangular_prism_l3471_347157


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3471_347158

def complex_number : ℂ := 2 + Complex.I

theorem complex_number_in_first_quadrant :
  Complex.re complex_number > 0 ∧ Complex.im complex_number > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3471_347158


namespace NUMINAMATH_CALUDE_specific_grid_has_nine_triangles_l3471_347118

/-- Represents the structure of the triangular grid with an additional triangle -/
structure TriangularGrid :=
  (bottom_row : Nat)
  (middle_row : Nat)
  (top_row : Nat)
  (additional : Nat)

/-- Counts the total number of triangles in the given grid structure -/
def count_triangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the specific grid structure has 9 triangles in total -/
theorem specific_grid_has_nine_triangles :
  let grid := TriangularGrid.mk 3 2 1 1
  count_triangles grid = 9 :=
by sorry

end NUMINAMATH_CALUDE_specific_grid_has_nine_triangles_l3471_347118


namespace NUMINAMATH_CALUDE_value_of_expression_l3471_347186

theorem value_of_expression (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3471_347186


namespace NUMINAMATH_CALUDE_cyclic_permutation_sum_equality_l3471_347138

def is_cyclic_shift (a : Fin n → ℕ) : Prop :=
  ∃ i, ∀ j, a j = ((j.val + i - 1) % n) + 1

def is_permutation (b : Fin n → ℕ) : Prop :=
  Function.Bijective b ∧ ∀ i, b i ≤ n

theorem cyclic_permutation_sum_equality (n : ℕ) :
  (∃ (a b : Fin n → ℕ),
    is_cyclic_shift a ∧
    is_permutation b ∧
    ∀ i j : Fin n, i.val + 1 + a i + b i = j.val + 1 + a j + b j) ↔
  Odd n :=
sorry

end NUMINAMATH_CALUDE_cyclic_permutation_sum_equality_l3471_347138


namespace NUMINAMATH_CALUDE_max_intersection_quadrilateral_pentagon_l3471_347153

/-- A polygon in the plane -/
structure Polygon :=
  (sides : ℕ)

/-- The number of intersection points between two polygons -/
def intersection_points (p1 p2 : Polygon) : ℕ := sorry

theorem max_intersection_quadrilateral_pentagon :
  ∃ (quad pent : Polygon),
    quad.sides = 4 ∧
    pent.sides = 5 ∧
    (∀ (q p : Polygon), q.sides = 4 → p.sides = 5 →
      intersection_points q p ≤ intersection_points quad pent) ∧
    intersection_points quad pent = 20 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_quadrilateral_pentagon_l3471_347153


namespace NUMINAMATH_CALUDE_correct_statements_l3471_347111

theorem correct_statements :
  (abs (-5) = 5) ∧ (-(- 3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l3471_347111


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l3471_347145

theorem greatest_integer_problem : 
  (∃ m : ℕ, 
    0 < m ∧ 
    m < 150 ∧ 
    (∃ a : ℤ, m = 10 * a - 2) ∧ 
    (∃ b : ℤ, m = 9 * b - 4) ∧
    (∀ n : ℕ, 
      (0 < n ∧ 
       n < 150 ∧ 
       (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
       (∃ b' : ℤ, n = 9 * b' - 4)) → 
      n ≤ m)) ∧
  (∀ m : ℕ, 
    (0 < m ∧ 
     m < 150 ∧ 
     (∃ a : ℤ, m = 10 * a - 2) ∧ 
     (∃ b : ℤ, m = 9 * b - 4) ∧
     (∀ n : ℕ, 
       (0 < n ∧ 
        n < 150 ∧ 
        (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
        (∃ b' : ℤ, n = 9 * b' - 4)) → 
       n ≤ m)) → 
    m = 68) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l3471_347145


namespace NUMINAMATH_CALUDE_surfer_ratio_l3471_347101

/-- Proves that the ratio of surfers on Malibu beach to Santa Monica beach is 2:1 -/
theorem surfer_ratio :
  ∀ (malibu santa_monica : ℕ),
  santa_monica = 20 →
  malibu + santa_monica = 60 →
  (malibu : ℚ) / santa_monica = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_surfer_ratio_l3471_347101


namespace NUMINAMATH_CALUDE_range_of_a_l3471_347189

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3471_347189


namespace NUMINAMATH_CALUDE_average_after_removing_two_l3471_347184

def initial_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,12]

def remove_element (list : List ℕ) (elem : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ elem)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_after_removing_two :
  average (remove_element initial_list 2) = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_average_after_removing_two_l3471_347184


namespace NUMINAMATH_CALUDE_complementary_not_supplementary_l3471_347142

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Theorem: It is impossible for two angles to be both complementary and supplementary -/
theorem complementary_not_supplementary : ¬ ∃ (a b : ℝ), complementary a b ∧ supplementary a b := by
  sorry

end NUMINAMATH_CALUDE_complementary_not_supplementary_l3471_347142


namespace NUMINAMATH_CALUDE_hexagon_walk_l3471_347160

/-- A regular hexagon with side length 3 km -/
structure RegularHexagon where
  sideLength : ℝ
  is_regular : sideLength = 3

/-- A point on the perimeter of the hexagon, represented by the distance traveled from a corner -/
def PerimeterPoint (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem hexagon_walk (h : RegularHexagon) :
  let start := (0, 0)
  let end_point := PerimeterPoint h 8
  distance start end_point = 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_walk_l3471_347160


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3471_347102

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_1 (n : ℕ) (d : ℕ) : Prop := n % d = 1

theorem smallest_number_satisfying_conditions : 
  (∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 6721 d) ∧ 
  is_divisible_by_11 6721 ∧
  (∀ m : ℕ, m < 6721 → 
    (¬(∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 m d) ∨ 
     ¬(is_divisible_by_11 m))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3471_347102


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l3471_347171

theorem quadratic_equation_transformation (x : ℝ) :
  (4 * x^2 + 8 * x - 468 = 0) →
  ∃ p q : ℝ, ((x + p)^2 = q) ∧ (q = 116) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l3471_347171


namespace NUMINAMATH_CALUDE_seventeen_doors_max_attempts_l3471_347130

/-- The maximum number of attempts needed to open n doors with n keys --/
def max_attempts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 17 doors and 17 keys, the maximum number of attempts is 136 --/
theorem seventeen_doors_max_attempts :
  max_attempts 17 = 136 := by sorry

end NUMINAMATH_CALUDE_seventeen_doors_max_attempts_l3471_347130


namespace NUMINAMATH_CALUDE_abby_coins_l3471_347180

theorem abby_coins (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 23)
  (h_total_value : total_value = 455/100)
  : ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    (25 * quarters + 5 * nickels : ℚ) / 100 = total_value ∧
    quarters = 17 := by
  sorry

end NUMINAMATH_CALUDE_abby_coins_l3471_347180


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3471_347104

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3471_347104


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l3471_347155

/-- Given a rectangular roof with length 4 times its width and an area of 900 square feet,
    prove that the difference between the length and width is 24√5 feet. -/
theorem roof_length_width_difference (w : ℝ) (h1 : w > 0) (h2 : 5 * w * w = 900) :
  5 * w - w = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_roof_length_width_difference_l3471_347155


namespace NUMINAMATH_CALUDE_triangle_area_l3471_347179

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the tangent line at (1, 1)
def tangent_line (x : ℝ) : ℝ := 3*x - 2

-- Define the x-axis (y = 0)
def x_axis (x : ℝ) : ℝ := 0

-- Define the vertical line x = 2
def vertical_line : ℝ := 2

-- Theorem statement
theorem triangle_area : 
  let x_intercept : ℝ := 2/3
  let height : ℝ := tangent_line vertical_line
  (1/2) * (vertical_line - x_intercept) * height = 8/3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3471_347179


namespace NUMINAMATH_CALUDE_sin_beta_value_l3471_347197

theorem sin_beta_value (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -(3/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3471_347197


namespace NUMINAMATH_CALUDE_blue_candies_count_l3471_347164

/-- The number of blue candies in a bag, given the following conditions:
    - There are 5 green candies and 4 red candies
    - The probability of picking a blue candy is 25% -/
def num_blue_candies : ℕ :=
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  3

theorem blue_candies_count :
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  let total_candies : ℕ := green_candies + red_candies + num_blue_candies
  (num_blue_candies : ℚ) / total_candies = prob_blue :=
by sorry

end NUMINAMATH_CALUDE_blue_candies_count_l3471_347164


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_2017_power_plus_2017_l3471_347141

theorem infinite_primes_dividing_2017_power_plus_2017 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2017^(2^n) + 2017} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_2017_power_plus_2017_l3471_347141


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l3471_347120

theorem polynomial_factor_implies_c_value : ∀ (c q : ℝ),
  (∃ (a : ℝ), (X^3 + q*X + 1) * (3*X + a) = 3*X^4 + c*X^2 + 8*X + 9) →
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l3471_347120


namespace NUMINAMATH_CALUDE_queue_probabilities_l3471_347140

/-- Probabilities of different numbers of people queuing -/
structure QueueProbabilities where
  p0 : ℝ  -- Probability of 0 people
  p1 : ℝ  -- Probability of 1 person
  p2 : ℝ  -- Probability of 2 people
  p3 : ℝ  -- Probability of 3 people
  p4 : ℝ  -- Probability of 4 people
  p5 : ℝ  -- Probability of 5 or more people
  sum_to_one : p0 + p1 + p2 + p3 + p4 + p5 = 1
  all_nonneg : 0 ≤ p0 ∧ 0 ≤ p1 ∧ 0 ≤ p2 ∧ 0 ≤ p3 ∧ 0 ≤ p4 ∧ 0 ≤ p5

/-- The probabilities for the specific scenario -/
def scenario : QueueProbabilities where
  p0 := 0.1
  p1 := 0.16
  p2 := 0.3
  p3 := 0.3
  p4 := 0.1
  p5 := 0.04
  sum_to_one := by sorry
  all_nonneg := by sorry

theorem queue_probabilities (q : QueueProbabilities) :
  (q.p0 + q.p1 + q.p2 = 0.56) ∧ 
  (q.p3 + q.p4 + q.p5 = 0.44) :=
by sorry

end NUMINAMATH_CALUDE_queue_probabilities_l3471_347140


namespace NUMINAMATH_CALUDE_lost_card_number_l3471_347115

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
sorry

end NUMINAMATH_CALUDE_lost_card_number_l3471_347115


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3471_347178

theorem geometric_progression_proof (b : ℕ → ℚ) 
  (h1 : b 4 - b 2 = -45/32) 
  (h2 : b 6 - b 4 = -45/512) 
  (h_geom : ∀ n : ℕ, b (n + 1) = b 1 * (b 2 / b 1) ^ n) :
  ((b 1 = -6 ∧ b 2 / b 1 = -1/4) ∨ (b 1 = 6 ∧ b 2 / b 1 = 1/4)) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3471_347178


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l3471_347163

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 886 / 1000)
  : (total_bananas - (good_fruits_percentage * (total_oranges + total_bananas) - (1 - rotten_oranges_percentage) * total_oranges)) / total_bananas = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l3471_347163


namespace NUMINAMATH_CALUDE_linear_regression_passes_through_mean_l3471_347198

variables {x y : ℝ} (x_bar y_bar a_hat b_hat : ℝ)

/-- The linear regression equation -/
def linear_regression (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The intercept of the linear regression equation -/
def intercept : ℝ := y_bar - b_hat * x_bar

theorem linear_regression_passes_through_mean :
  a_hat = intercept x_bar y_bar b_hat →
  linear_regression x_bar a_hat b_hat = y_bar :=
sorry

end NUMINAMATH_CALUDE_linear_regression_passes_through_mean_l3471_347198


namespace NUMINAMATH_CALUDE_mycoplasma_pneumonia_relation_l3471_347199

-- Define the contingency table
def a : ℕ := 40  -- infected with mycoplasma pneumonia and with chronic disease
def b : ℕ := 20  -- infected with mycoplasma pneumonia and without chronic disease
def c : ℕ := 60  -- not infected with mycoplasma pneumonia and with chronic disease
def d : ℕ := 80  -- not infected with mycoplasma pneumonia and without chronic disease
def n : ℕ := a + b + c + d

-- Define the K^2 statistic
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.5% confidence level
def critical_value : ℚ := 7.879

-- Define the number of cases with exactly one person having chronic disease
def favorable_cases : ℕ := 8
def total_cases : ℕ := 15

theorem mycoplasma_pneumonia_relation :
  K_squared > critical_value ∧ (favorable_cases : ℚ) / total_cases = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mycoplasma_pneumonia_relation_l3471_347199


namespace NUMINAMATH_CALUDE_ms_elizabeth_has_five_investments_l3471_347108

-- Define the variables
def mr_banks_investments : ℕ := 8
def mr_banks_revenue_per_investment : ℕ := 500
def ms_elizabeth_revenue_per_investment : ℕ := 900
def revenue_difference : ℕ := 500

-- Define Ms. Elizabeth's number of investments as a function
def ms_elizabeth_investments : ℕ :=
  let mr_banks_total_revenue := mr_banks_investments * mr_banks_revenue_per_investment
  let ms_elizabeth_total_revenue := mr_banks_total_revenue + revenue_difference
  ms_elizabeth_total_revenue / ms_elizabeth_revenue_per_investment

-- Theorem statement
theorem ms_elizabeth_has_five_investments :
  ms_elizabeth_investments = 5 := by
  sorry

end NUMINAMATH_CALUDE_ms_elizabeth_has_five_investments_l3471_347108


namespace NUMINAMATH_CALUDE_percent_y_of_x_l3471_347187

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y / x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l3471_347187


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3471_347150

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 5 * (a₂ - b₂) →
  p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3471_347150


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3471_347183

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3471_347183


namespace NUMINAMATH_CALUDE_inequalities_hold_l3471_347119

theorem inequalities_hold (m n l : ℝ) (h1 : m > n) (h2 : n > l) : 
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3471_347119


namespace NUMINAMATH_CALUDE_proper_divisor_cube_difference_l3471_347109

theorem proper_divisor_cube_difference (n : ℕ) : 
  (∃ (x y : ℕ), 
    x > 1 ∧ y > 1 ∧
    x ∣ n ∧ y ∣ n ∧
    n ≠ x ∧ n ≠ y ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≥ x) ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≤ y) ∧
    (y = x^3 + 3 ∨ y = x^3 - 3)) ↔
  (n = 10 ∨ n = 22) :=
sorry

end NUMINAMATH_CALUDE_proper_divisor_cube_difference_l3471_347109


namespace NUMINAMATH_CALUDE_square_on_parabola_diagonal_l3471_347131

/-- Given a square ABOC where O is the origin, A and B are on the parabola y = -x^2,
    and C is opposite to O, the length of diagonal AC is 2a, where a is the x-coordinate of point A. -/
theorem square_on_parabola_diagonal (a : ℝ) :
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (a, a^2)
  -- ABOC is a square
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = (O.1 - C.1)^2 + (O.2 - C.2)^2 ∧
  (O.1 - C.1)^2 + (O.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  -- Length of AC is 2a
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2*a)^2 :=
by sorry


end NUMINAMATH_CALUDE_square_on_parabola_diagonal_l3471_347131


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l3471_347105

theorem unique_solution_trig_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (2 * n.val) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l3471_347105


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3471_347143

theorem arithmetic_expression_equality : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3471_347143


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3471_347114

theorem quadratic_root_problem (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + m*x - 6
  f (-6) = 0 → ∃ (x : ℝ), x ≠ -6 ∧ f x = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3471_347114


namespace NUMINAMATH_CALUDE_power_division_l3471_347177

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_l3471_347177


namespace NUMINAMATH_CALUDE_last_five_shots_made_l3471_347191

/-- Represents the number of shots made in a series of basketball attempts -/
structure ShotsMade where
  total : ℕ
  made : ℕ

/-- Calculates the shooting percentage -/
def shootingPercentage (s : ShotsMade) : ℚ :=
  s.made / s.total

theorem last_five_shots_made
  (initial : ShotsMade)
  (second : ShotsMade)
  (final : ShotsMade)
  (h1 : initial.total = 30)
  (h2 : shootingPercentage initial = 2/5)
  (h3 : second.total = initial.total + 10)
  (h4 : shootingPercentage second = 9/20)
  (h5 : final.total = second.total + 5)
  (h6 : shootingPercentage final = 23/50)
  : final.made - second.made = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_five_shots_made_l3471_347191


namespace NUMINAMATH_CALUDE_sequence_length_l3471_347136

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem sequence_length : 
  ∃ n : ℕ, n > 0 ∧ 
  arithmetic_sequence 2.5 5 n = 62.5 ∧ 
  ∀ k : ℕ, k > n → arithmetic_sequence 2.5 5 k > 62.5 ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l3471_347136


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_specific_evaluation_l3471_347161

theorem expression_simplification_and_evaluation (a b : ℚ) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = 4*a*b :=
by sorry

theorem specific_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_specific_evaluation_l3471_347161
