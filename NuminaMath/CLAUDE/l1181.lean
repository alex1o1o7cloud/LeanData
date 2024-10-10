import Mathlib

namespace sum_of_three_squares_squared_l1181_118105

theorem sum_of_three_squares_squared (a b c : ℕ) :
  ∃ (x y z : ℕ), (a^2 + b^2 + c^2)^2 = x^2 + y^2 + z^2 := by
  sorry

end sum_of_three_squares_squared_l1181_118105


namespace contrapositive_prop2_true_l1181_118188

theorem contrapositive_prop2_true : 
  (∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0)) := by sorry

end contrapositive_prop2_true_l1181_118188


namespace two_numbers_difference_l1181_118165

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff_eq : x^2 - y^2 = 40) : 
  |x - y| = 4 := by
sorry

end two_numbers_difference_l1181_118165


namespace cone_surface_area_l1181_118175

/-- A cone with slant height 2 and lateral surface unfolding into a semicircle has surface area 3π -/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 → -- slant height is 2
  2 * π * r = 2 * π → -- lateral surface unfolds into a semicircle (circumference of base equals arc length of semicircle)
  π * r * (r + h) = 3 * π := by
  sorry


end cone_surface_area_l1181_118175


namespace quadratic_origin_condition_l1181_118179

/-- A quadratic function passing through the origin -/
def passes_through_origin (m : ℝ) : Prop :=
  ∃ x y : ℝ, y = m * x^2 + x + m * (m - 2) ∧ x = 0 ∧ y = 0

/-- The theorem stating the conditions for the quadratic function to pass through the origin -/
theorem quadratic_origin_condition :
  ∀ m : ℝ, passes_through_origin m ↔ m = 2 ∨ m = 0 := by
  sorry

end quadratic_origin_condition_l1181_118179


namespace micahs_strawberries_l1181_118109

def strawberries_for_mom (picked : ℕ) (eaten : ℕ) : ℕ :=
  picked - eaten

theorem micahs_strawberries :
  strawberries_for_mom (2 * 12) 6 = 18 := by
  sorry

end micahs_strawberries_l1181_118109


namespace quadratic_equation_solution_l1181_118135

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end quadratic_equation_solution_l1181_118135


namespace bales_in_barn_l1181_118156

/-- The number of bales in the barn after Tim added a couple more -/
def total_bales (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- A couple is defined as 2 -/
def couple : ℕ := 2

theorem bales_in_barn (initial_bales : ℕ) (h : initial_bales = 540) :
  total_bales initial_bales couple = 542 := by
  sorry

end bales_in_barn_l1181_118156


namespace derivative_at_one_l1181_118195

open Real

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∀ x, f x = 2 * x * f' 1 + log x) →
  f' 1 = -1 := by
sorry

end derivative_at_one_l1181_118195


namespace min_max_of_expression_l1181_118147

open Real

theorem min_max_of_expression (x : ℝ) (h : x > 2) :
  let f := fun x => (x + 9) / sqrt (x - 2)
  (∃ (m : ℝ), m = 2 * sqrt 11 ∧ 
    (∀ y, y > 2 → f y ≥ m) ∧ 
    f 13 = m) ∧
  (∀ M : ℝ, ∃ y, y > 2 ∧ f y > M) :=
by sorry

end min_max_of_expression_l1181_118147


namespace decimal_73_is_four_digits_in_base_4_l1181_118126

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem stating that 73 in decimal is a four-digit number in base 4 -/
theorem decimal_73_is_four_digits_in_base_4 :
  (toBase4 73).length = 4 :=
sorry

end decimal_73_is_four_digits_in_base_4_l1181_118126


namespace log_equality_difference_l1181_118196

theorem log_equality_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = (Real.log d) / (Real.log c))
  (h2 : a - c = 9) : 
  b - d = 93 := by sorry

end log_equality_difference_l1181_118196


namespace alicia_tax_deduction_l1181_118140

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 18 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

/-- Theorem stating that Alicia's tax deduction is 45 cents per hour -/
theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 45 := by
  sorry

end alicia_tax_deduction_l1181_118140


namespace pipe_filling_time_l1181_118159

theorem pipe_filling_time (fill_rate_A fill_rate_B fill_rate_C : ℝ) 
  (h1 : fill_rate_A + fill_rate_B + fill_rate_C = 1 / 5)
  (h2 : fill_rate_B = 2 * fill_rate_A)
  (h3 : fill_rate_C = 2 * fill_rate_B) :
  1 / fill_rate_A = 35 := by
  sorry

end pipe_filling_time_l1181_118159


namespace cube_sum_reciprocal_l1181_118194

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_reciprocal_l1181_118194


namespace polygon_interior_angles_l1181_118185

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by
sorry

end polygon_interior_angles_l1181_118185


namespace average_fish_caught_l1181_118150

def aang_fish : ℕ := 7
def sokka_fish : ℕ := 5
def toph_fish : ℕ := 12
def total_people : ℕ := 3

theorem average_fish_caught :
  (aang_fish + sokka_fish + toph_fish) / total_people = 8 := by
  sorry

end average_fish_caught_l1181_118150


namespace black_area_from_white_area_l1181_118173

/-- Represents a square divided into 9 equal smaller squares -/
structure DividedSquare where
  total_area : ℝ
  white_squares : ℕ
  black_squares : ℕ
  white_area : ℝ

/-- Theorem stating the relation between white and black areas in the divided square -/
theorem black_area_from_white_area (s : DividedSquare) 
  (h1 : s.white_squares + s.black_squares = 9)
  (h2 : s.white_squares = 5)
  (h3 : s.black_squares = 4)
  (h4 : s.white_area = 180) :
  s.total_area * (s.black_squares / 9 : ℝ) = 144 := by
sorry

end black_area_from_white_area_l1181_118173


namespace center_sum_is_seven_l1181_118133

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y - 15

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))

theorem center_sum_is_seven :
  ∃ h k, CircleCenter h k CircleEquation ∧ h + k = 7 := by
  sorry

end center_sum_is_seven_l1181_118133


namespace complex_subtraction_l1181_118121

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I :=
by sorry

end complex_subtraction_l1181_118121


namespace existence_of_sequence_l1181_118113

theorem existence_of_sequence (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ a : Fin (n + 1) → ℝ,
    (a 0 + a (Fin.last n) = 0) ∧
    (∀ i, |a i| ≤ 1) ∧
    (∀ i : Fin n, |a i.succ - a i| = x i) := by
  sorry

end existence_of_sequence_l1181_118113


namespace range_of_a_l1181_118189

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ 
    (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, condition_M a ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l1181_118189


namespace lost_card_sum_l1181_118157

theorem lost_card_sum (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  (∃ (s : Finset ℝ), s ⊆ sums.toFinset ∧ s.card = 5 ∧ 
    (270 ∈ s ∧ 360 ∈ s ∧ 390 ∈ s ∧ 500 ∈ s ∧ 620 ∈ s)) →
  530 ∈ sums.toFinset :=
by sorry

end lost_card_sum_l1181_118157


namespace positive_expression_l1181_118111

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c) + a * (b^2 + c^2 - b*c) > 0 := by
  sorry

end positive_expression_l1181_118111


namespace special_collection_total_l1181_118138

/-- A collection of shapes consisting of circles, squares, and triangles. -/
structure ShapeCollection where
  circles : ℕ
  squares : ℕ
  triangles : ℕ

/-- The total number of shapes in the collection. -/
def ShapeCollection.total (sc : ShapeCollection) : ℕ :=
  sc.circles + sc.squares + sc.triangles

/-- A collection satisfying the given conditions. -/
def specialCollection : ShapeCollection :=
  { circles := 5, squares := 1, triangles := 9 }

theorem special_collection_total :
  (specialCollection.squares + specialCollection.triangles = 10) ∧
  (specialCollection.circles + specialCollection.triangles = 14) ∧
  (specialCollection.circles + specialCollection.squares = 6) ∧
  specialCollection.total = 15 := by
  sorry

#eval specialCollection.total

end special_collection_total_l1181_118138


namespace m_n_properties_l1181_118136

theorem m_n_properties (m n : ℤ) (hm : |m| = 1) (hn : |n| = 4) :
  (∃ k : ℤ, mn < 0 → m + n = k ∧ (k = 3 ∨ k = -3)) ∧
  (∀ x y : ℤ, |x| = 1 → |y| = 4 → x - y ≤ 5) ∧
  (∃ a b : ℤ, |a| = 1 ∧ |b| = 4 ∧ a - b = 5) :=
by sorry

end m_n_properties_l1181_118136


namespace light_reflection_l1181_118199

/-- A beam of light passing through a point and reflecting off a circle --/
structure LightBeam where
  M : ℝ × ℝ
  C : Set (ℝ × ℝ)

/-- Definition of the circle C --/
def is_circle (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 7)^2 = 25

/-- Definition of the reflected light ray equation --/
def reflected_ray_equation (x y : ℝ) : Prop :=
  x + y - 7 = 0

/-- Definition of the range of incident point A --/
def incident_point_range (A : ℝ) : Prop :=
  1 ≤ A ∧ A ≤ 23/2

/-- Main theorem --/
theorem light_reflection (beam : LightBeam) 
  (h_M : beam.M = (25, 18))
  (h_C : is_circle beam.C) :
  (∀ x y, reflected_ray_equation x y ↔ 
    (x, y) ∈ {p | ∃ t, p = ((1-t) * 25 + t * 0, (1-t) * (-18) + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) ∧
  (∀ A, incident_point_range A ↔ 
    ∃ (k : ℝ), (A, 0) ∈ {p | ∃ t, p = ((1-t) * 25 + t * A, (1-t) * (-18) + t * 0) ∧ 0 ≤ t ∧ t ≤ 1} ∧
               (0, 7) ∈ {p | ∃ t, p = ((1-t) * A + t * 0, (1-t) * 0 + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) :=
sorry

end light_reflection_l1181_118199


namespace pencil_distribution_l1181_118127

/-- The number of ways to distribute n identical objects among k people,
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 9

/-- Each friend must have at least one pencil -/
def min_pencils_per_friend : ℕ := 1

theorem pencil_distribution :
  distribute (total_pencils - num_friends * min_pencils_per_friend + num_friends) num_friends = 28 := by
  sorry

end pencil_distribution_l1181_118127


namespace no_intersection_quadratic_sets_l1181_118161

theorem no_intersection_quadratic_sets (A B : ℤ) :
  ∃ C : ℤ, ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C := by
  sorry

end no_intersection_quadratic_sets_l1181_118161


namespace line_passes_through_point_l1181_118112

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t m x y : ℝ) : Prop := x = t*y + m

-- Define point P
def point_P : ℝ × ℝ := (-2, 0)

-- Define the condition that l is not vertical to x-axis
def not_vertical (t : ℝ) : Prop := t ≠ 0

-- Define the bisection condition
def bisects_angle (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  y₁ / (x₁ + 2) + y₂ / (x₂ + 2) = 0

-- Main theorem
theorem line_passes_through_point :
  ∀ (t m : ℝ) (A B : ℝ × ℝ),
  not_vertical t →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line_l t m A.1 A.2 →
  line_l t m B.1 B.2 →
  A ≠ B →
  bisects_angle A B →
  ∃ (x : ℝ), line_l t m x 0 ∧ x = 2 :=
sorry

end line_passes_through_point_l1181_118112


namespace f_properties_l1181_118101

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 4| - 3

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≠ -2) :
  (f a a > f a (-2)) ∧
  (∃ x y : ℝ, x < y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z ∈ Set.Ioo x y, f a z > 0) ↔
  a ∈ Set.Ioc (-5) (-7/2) ∪ Set.Ico (-1/2) 1 :=
sorry

end f_properties_l1181_118101


namespace raccoon_lock_problem_l1181_118167

theorem raccoon_lock_problem (first_lock_duration second_lock_duration : ℕ) : 
  first_lock_duration = 5 →
  second_lock_duration < 3 * first_lock_duration →
  5 * second_lock_duration = 60 →
  3 * first_lock_duration - second_lock_duration = 3 :=
by
  sorry

end raccoon_lock_problem_l1181_118167


namespace seventh_term_is_seven_l1181_118191

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Fourth term is 4
  fourth_term : a + 3*d = 4

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end seventh_term_is_seven_l1181_118191


namespace complex_square_in_fourth_quadrant_l1181_118197

theorem complex_square_in_fourth_quadrant (z : ℂ) :
  (z.re > 0 ∧ z.im < 0) →  -- z is in the fourth quadrant
  z^2 - 2*z + 2 = 0 →      -- z satisfies the given equation
  z^2 = -2*Complex.I :=    -- conclusion: z^2 = -2i
by
  sorry

end complex_square_in_fourth_quadrant_l1181_118197


namespace original_number_l1181_118144

theorem original_number (x : ℚ) (h : 1 - 1/x = 5/2) : x = -2/3 := by
  sorry

end original_number_l1181_118144


namespace f_property_and_g_monotonicity_l1181_118172

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + x + 1

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := fun x ↦ |f x - a * x + 3|

theorem f_property_and_g_monotonicity :
  (∀ x : ℝ, f (1 - x) = x^2 - 3*x + 3) ∧
  (∀ x : ℝ, f x = x^2 + x + 1) ∧
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → g a x ≤ g a y) ↔ 
    (a ≤ 3 ∨ 6 ≤ a)) :=
by sorry

end f_property_and_g_monotonicity_l1181_118172


namespace intersection_with_complement_l1181_118184

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end intersection_with_complement_l1181_118184


namespace cat_hunting_theorem_l1181_118110

/-- The number of birds caught during the day -/
def day_birds : ℕ := 8

/-- The number of birds caught at night -/
def night_birds : ℕ := 2 * day_birds

/-- The total number of birds caught -/
def total_birds : ℕ := 24

theorem cat_hunting_theorem : 
  day_birds + night_birds = total_birds ∧ night_birds = 2 * day_birds :=
by sorry

end cat_hunting_theorem_l1181_118110


namespace nabla_properties_and_equation_solution_l1181_118139

-- Define the ∇ operation
def nabla (X Y : ℤ) : ℤ := X + 2 * Y

-- State the theorem
theorem nabla_properties_and_equation_solution :
  (∀ X : ℤ, nabla X 0 = X) ∧
  (∀ X Y : ℤ, nabla X (Y - 1) = nabla X Y - 2) ∧
  (∀ X Y : ℤ, nabla X (Y + 1) = nabla X Y + 2) →
  (∀ X Y : ℤ, nabla X Y = X + 2 * Y) ∧
  (nabla (-673) (-673) = -2019 ∧ ∀ X : ℤ, nabla X X = -2019 → X = -673) :=
by sorry

end nabla_properties_and_equation_solution_l1181_118139


namespace equation_solution_system_solution_l1181_118183

-- Define the equation
def equation (x : ℚ) : Prop := 64 * (x - 1)^3 + 27 = 0

-- Define the system of equations
def system (x y : ℚ) : Prop := x + y = 3 ∧ 2*x - 3*y = 6

-- Theorem for the equation
theorem equation_solution : ∃ x : ℚ, equation x ∧ x = 1/4 := by sorry

-- Theorem for the system of equations
theorem system_solution : ∃ x y : ℚ, system x y ∧ x = 3 ∧ y = 0 := by sorry

end equation_solution_system_solution_l1181_118183


namespace complex_magnitude_l1181_118117

theorem complex_magnitude (z : ℂ) (h : z = 4 + 3 * I) : Complex.abs z = 5 := by
  sorry

end complex_magnitude_l1181_118117


namespace mutual_choice_exists_l1181_118118

/-- A monotonic increasing function from {1,...,n} to {1,...,n} -/
def MonotonicFunction (n : ℕ) := {f : Fin n → Fin n // ∀ i j, i ≤ j → f i ≤ f j}

/-- The theorem statement -/
theorem mutual_choice_exists (n : ℕ) (hn : n > 0) (f g : MonotonicFunction n) :
  ∃ k : Fin n, (f.val ∘ g.val) k = k :=
sorry

end mutual_choice_exists_l1181_118118


namespace negation_of_at_most_two_l1181_118129

theorem negation_of_at_most_two (P : ℕ → Prop) : 
  (¬ (∃ n : ℕ, P n ∧ (∀ m : ℕ, P m → m ≤ n) ∧ n ≤ 2)) ↔ 
  (∃ a b c : ℕ, P a ∧ P b ∧ P c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end negation_of_at_most_two_l1181_118129


namespace range_of_f_l1181_118106

noncomputable def f (x : ℝ) : ℝ := 1 - x - 9 / x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 0 ∧ f x = y) ↔ y ≤ -5 ∨ y ≥ 7 := by
  sorry

end range_of_f_l1181_118106


namespace base9_sequence_is_triangular_l1181_118134

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Definition of the sequence in base-9 -/
def base9_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 9 * base9_sequence n + 1

/-- Theorem stating that each term in the base-9 sequence is a triangular number -/
theorem base9_sequence_is_triangular (n : ℕ) : 
  ∃ m : ℕ, base9_sequence n = triangular m := by sorry

end base9_sequence_is_triangular_l1181_118134


namespace largest_last_digit_l1181_118169

def is_valid_series (s : List Nat) : Prop :=
  s.length = 2023 ∧
  s.head? = some 1 ∧
  ∀ i, i < s.length - 1 →
    let two_digit := s[i]! * 10 + s[i+1]!
    two_digit % 17 = 0 ∨ two_digit % 29 = 0 ∨ two_digit % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_series s) :
  s.getLast? = some 2 := by
  sorry

end largest_last_digit_l1181_118169


namespace polar_to_rectangular_equivalence_l1181_118123

/-- The polar coordinate equation ρ = 4sin(θ) + 2cos(θ) is equivalent to
    the rectangular coordinate equation (x-1)^2 + (y-2)^2 = 5 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  ρ = 4 * Real.sin θ + 2 * Real.cos θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (x - 1)^2 + (y - 2)^2 = 5 := by
sorry

end polar_to_rectangular_equivalence_l1181_118123


namespace zachary_needs_additional_money_l1181_118162

def football_cost : ℚ := 3.75
def shorts_cost : ℚ := 2.40
def shoes_cost : ℚ := 11.85
def zachary_money : ℚ := 10.00

theorem zachary_needs_additional_money :
  football_cost + shorts_cost + shoes_cost - zachary_money = 7.00 := by
  sorry

end zachary_needs_additional_money_l1181_118162


namespace arithmetic_sequence_common_difference_l1181_118125

theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (sum : ℚ) 
  (h₁ : a₁ = 3) 
  (h₂ : aₙ = 50) 
  (h₃ : sum = 318) : 
  ∃ (n : ℕ) (d : ℚ), 
    n > 1 ∧ 
    aₙ = a₁ + (n - 1) * d ∧ 
    sum = (n / 2) * (a₁ + aₙ) ∧ 
    d = 47 / 11 := by
  sorry

end arithmetic_sequence_common_difference_l1181_118125


namespace problem_statement_l1181_118176

theorem problem_statement (a b c : ℝ) 
  (h_diff1 : a ≠ b) (h_diff2 : b ≠ c) (h_diff3 : a ≠ c)
  (h_eq : Real.sqrt (a^3 * (b-a)^3) - Real.sqrt (a^3 * (c-a)^3) = Real.sqrt (a-b) - Real.sqrt (c-a)) :
  a^2 + b^2 + c^2 - 2*a*b + 2*b*c - 2*a*c = 0 := by
sorry

end problem_statement_l1181_118176


namespace cube_of_product_l1181_118104

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end cube_of_product_l1181_118104


namespace quadratic_roots_nature_l1181_118107

theorem quadratic_roots_nature (a b c m n : ℝ) : 
  a ≠ 0 → c ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m * n < 0 →
  m < abs m →
  ∀ x, c * x^2 + (m - n) * a * x - a = 0 → x < 0 :=
sorry

end quadratic_roots_nature_l1181_118107


namespace company_employees_l1181_118146

theorem company_employees (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  wednesday_birthdays = 13 →
  wednesday_birthdays > other_day_birthdays →
  (7 * other_day_birthdays + wednesday_birthdays - other_day_birthdays : ℕ) = 85 :=
by sorry

end company_employees_l1181_118146


namespace alice_outfits_l1181_118100

/-- The number of different outfits Alice can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Alice can create with her wardrobe -/
theorem alice_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end alice_outfits_l1181_118100


namespace extended_morse_code_symbols_l1181_118153

def morse_symbols (n : Nat) : Nat :=
  3^n

theorem extended_morse_code_symbols :
  (morse_symbols 1) + (morse_symbols 2) + (morse_symbols 3) + (morse_symbols 4) = 120 := by
  sorry

end extended_morse_code_symbols_l1181_118153


namespace diophantine_equation_solution_l1181_118143

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x + y = z ∧ x^2 * y = z^2 + 1) →
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 13 ∧ z = 18)) :=
by
  sorry

end diophantine_equation_solution_l1181_118143


namespace number_fraction_theorem_l1181_118114

theorem number_fraction_theorem (number : ℚ) (fraction : ℚ) : 
  number = 64 →
  number = number * fraction + 40 →
  fraction = 3 / 8 := by
  sorry

end number_fraction_theorem_l1181_118114


namespace points_divisible_by_ten_l1181_118116

/-- A configuration of points on a circle satisfying certain distance conditions -/
structure PointConfiguration where
  n : ℕ
  circle_length : ℕ
  distance_one : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 1
  distance_two : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 2

/-- Theorem stating that for a specific configuration, n is divisible by 10 -/
theorem points_divisible_by_ten (config : PointConfiguration) 
  (h_length : config.circle_length = 15) : 
  10 ∣ config.n :=
sorry

end points_divisible_by_ten_l1181_118116


namespace power_of_eight_mod_five_l1181_118142

theorem power_of_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end power_of_eight_mod_five_l1181_118142


namespace complex_root_theorem_l1181_118192

theorem complex_root_theorem (z : ℂ) (p : ℝ) : 
  (z^2 + 2*z + p = 0) → (Complex.abs z = 2) → (p = 4) := by
  sorry

end complex_root_theorem_l1181_118192


namespace unique_perfect_square_sum_diff_l1181_118115

theorem unique_perfect_square_sum_diff (a : ℕ) : 
  (∃ b : ℕ, a * a = (b + 1) * (b + 1) - b * b ∧ 
            a * a = b * b + (b + 1) * (b + 1)) ∧ 
  a * a < 20000 ↔ 
  a = 1 :=
sorry

end unique_perfect_square_sum_diff_l1181_118115


namespace negative_integer_solution_exists_l1181_118186

theorem negative_integer_solution_exists : ∃ (x : ℤ), x < 0 ∧ 3 * x + 13 ≥ 0 :=
by
  -- Proof goes here
  sorry

end negative_integer_solution_exists_l1181_118186


namespace unused_signs_l1181_118103

theorem unused_signs (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) : 
  total_signs = 424 →
  used_signs = 422 →
  additional_codes = 1688 →
  total_signs ^ 2 - used_signs ^ 2 = additional_codes →
  total_signs - used_signs = 2 :=
by sorry

end unused_signs_l1181_118103


namespace factorial_multiple_implies_inequality_l1181_118168

theorem factorial_multiple_implies_inequality (a b : ℕ+) 
  (h : (a.val.factorial * b.val.factorial) % (a.val.factorial + b.val.factorial) = 0) : 
  3 * a.val ≥ 2 * b.val + 2 := by
  sorry

end factorial_multiple_implies_inequality_l1181_118168


namespace instantaneous_velocity_at_3s_l1181_118128

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 + 10

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s :
  v 3 = 6 :=
sorry

end instantaneous_velocity_at_3s_l1181_118128


namespace hyperbola_asymptotes_equation_l1181_118177

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a, b > 0,
    if its eccentricity e and the slope of its asymptotes k satisfy e = √2 |k|,
    then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let k := b / a
  e = Real.sqrt 2 * abs k →
  ∃ (f : ℝ → ℝ), (∀ x, f x = x ∨ f x = -x) ∧
    (∀ x y, y = f x ↔ (x^2 / a^2 - y^2 / b^2 = 1 → False)) :=
by sorry

end hyperbola_asymptotes_equation_l1181_118177


namespace largest_square_tile_l1181_118149

theorem largest_square_tile (board_length board_width tile_size : ℕ) : 
  board_length = 16 →
  board_width = 24 →
  tile_size = Nat.gcd board_length board_width →
  tile_size = 8 := by
sorry

end largest_square_tile_l1181_118149


namespace number_divided_by_three_l1181_118155

theorem number_divided_by_three : ∃ x : ℝ, (x / 3 = x - 48) ∧ (x = 72) := by
  sorry

end number_divided_by_three_l1181_118155


namespace water_tank_problem_l1181_118158

theorem water_tank_problem (c : ℝ) (h1 : c > 0) : 
  let w := c / 3
  let w' := w + 5
  let w'' := w' + 4
  (w / c = 1 / 3) ∧ (w' / c = 2 / 5) → w'' / c = 34 / 75 := by
sorry

end water_tank_problem_l1181_118158


namespace gcd_37_power_plus_one_l1181_118198

theorem gcd_37_power_plus_one (h : Prime 37) : 
  Nat.gcd (37^11 + 1) (37^11 + 37^3 + 1) = 1 := by
sorry

end gcd_37_power_plus_one_l1181_118198


namespace quadratic_equation_solution_l1181_118120

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 100 ↔ x = -10 ∨ x = 10 := by sorry

end quadratic_equation_solution_l1181_118120


namespace complex_number_problem_l1181_118164

theorem complex_number_problem :
  let z : ℂ := ((1 - I)^2 + 3*(1 + I)) / (2 - I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - I ∧ z = 1 + I ∧ a = -3 ∧ b = 4 := by
  sorry

end complex_number_problem_l1181_118164


namespace trig_equation_solution_l1181_118160

theorem trig_equation_solution (x : ℝ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 →
  ∃ k : ℤ, x = π * (2 * k + 1) := by
sorry

end trig_equation_solution_l1181_118160


namespace exponential_continuous_l1181_118132

/-- The exponential function is continuous for any positive base -/
theorem exponential_continuous (a : ℝ) (h : a > 0) :
  Continuous (fun x => a^x) :=
by
  sorry

end exponential_continuous_l1181_118132


namespace crate_stacking_probability_l1181_118163

def crate_height : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

def total_combinations : ℕ := 3^10

def valid_combinations : ℕ := 2940

theorem crate_stacking_probability :
  (valid_combinations : ℚ) / total_combinations = 980 / 19683 :=
sorry

end crate_stacking_probability_l1181_118163


namespace min_value_theorem_l1181_118130

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The set of available digits -/
def availableDigits : Finset Nat := {5, 5, 6, 6, 6, 7, 8, 8, 9}

/-- Theorem stating the minimum value of A + B - C -/
theorem min_value_theorem (A B C : ThreeDigitNumber) 
  (h1 : A.hundreds ∈ availableDigits)
  (h2 : A.tens ∈ availableDigits)
  (h3 : A.ones ∈ availableDigits)
  (h4 : B.hundreds ∈ availableDigits)
  (h5 : B.tens ∈ availableDigits)
  (h6 : B.ones ∈ availableDigits)
  (h7 : C.hundreds ∈ availableDigits)
  (h8 : C.tens ∈ availableDigits)
  (h9 : C.ones ∈ availableDigits)
  (h10 : A.toNat + B.toNat - C.toNat ≥ 149) :
  ∃ (A' B' C' : ThreeDigitNumber),
    A'.hundreds ∈ availableDigits ∧
    A'.tens ∈ availableDigits ∧
    A'.ones ∈ availableDigits ∧
    B'.hundreds ∈ availableDigits ∧
    B'.tens ∈ availableDigits ∧
    B'.ones ∈ availableDigits ∧
    C'.hundreds ∈ availableDigits ∧
    C'.tens ∈ availableDigits ∧
    C'.ones ∈ availableDigits ∧
    A'.toNat + B'.toNat - C'.toNat = 149 := by
  sorry

end min_value_theorem_l1181_118130


namespace dance_school_relation_l1181_118178

/-- Represents the dance school scenario -/
structure DanceSchool where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls

/-- The number of girls the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := 2 * n + 4

/-- The dance school satisfies the given conditions -/
def valid_dance_school (ds : DanceSchool) : Prop :=
  ∀ n, n ≥ 1 → n ≤ ds.b → girls_danced n ≤ ds.g ∧
  girls_danced ds.b = ds.g

theorem dance_school_relation (ds : DanceSchool) 
  (h : valid_dance_school ds) : 
  ds.b = (ds.g - 4) / 2 :=
sorry

end dance_school_relation_l1181_118178


namespace long_division_problem_l1181_118148

theorem long_division_problem (quotient remainder divisor dividend : ℕ) : 
  quotient = 2015 → 
  remainder = 0 → 
  divisor = 105 → 
  dividend = quotient * divisor + remainder → 
  dividend = 20685 := by
sorry

end long_division_problem_l1181_118148


namespace binomial_coefficient_28_7_l1181_118154

theorem binomial_coefficient_28_7 
  (h1 : Nat.choose 26 3 = 2600)
  (h2 : Nat.choose 26 4 = 14950)
  (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 := by
sorry

end binomial_coefficient_28_7_l1181_118154


namespace exam_pass_percentage_l1181_118190

/-- Given an examination where 740 students appeared and 481 failed,
    prove that 35% of students passed. -/
theorem exam_pass_percentage 
  (total_students : ℕ) 
  (failed_students : ℕ) 
  (h1 : total_students = 740)
  (h2 : failed_students = 481) : 
  (total_students - failed_students : ℚ) / total_students * 100 = 35 := by
sorry

end exam_pass_percentage_l1181_118190


namespace julien_contribution_julien_contribution_proof_l1181_118171

/-- The amount Julien needs to contribute to buy a pie -/
theorem julien_contribution (pie_cost : ℝ) (lucas_money : ℝ) (exchange_rate : ℝ) : ℝ :=
  pie_cost - lucas_money / exchange_rate

/-- Proof of Julien's required contribution -/
theorem julien_contribution_proof :
  julien_contribution 12 10 1.5 = 16 / 3 := by
  sorry

end julien_contribution_julien_contribution_proof_l1181_118171


namespace floor_tiling_l1181_118181

/-- 
Theorem: For an n × n floor to be completely covered by an equal number of 2 × 2 and 2 × 1 tiles, 
n must be divisible by 6.
-/
theorem floor_tiling (n : ℕ) : 
  (∃ x : ℕ, n * n = 6 * x) ↔ 6 ∣ n :=
by sorry

end floor_tiling_l1181_118181


namespace secretary_work_time_l1181_118124

theorem secretary_work_time (x y z : ℕ) (h1 : x + y + z = 80) (h2 : 2 * x = 3 * y) (h3 : 2 * x = z) : z = 40 := by
  sorry

end secretary_work_time_l1181_118124


namespace no_interior_projection_on_all_sides_l1181_118182

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Perpendicular projection of a point onto a line segment -/
def perp_projection (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a point is an interior point of a line segment -/
def is_interior (P A B : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem no_interior_projection_on_all_sides (t : Triangle) :
  ¬ ∃ P ∈ circumcircle t,
    (is_interior (perp_projection P t.A t.B) t.A t.B) ∧
    (is_interior (perp_projection P t.B t.C) t.B t.C) ∧
    (is_interior (perp_projection P t.C t.A) t.C t.A) :=
sorry

end no_interior_projection_on_all_sides_l1181_118182


namespace middle_school_running_average_middle_school_running_average_proof_l1181_118137

/-- The average number of minutes run per day by middle school students -/
theorem middle_school_running_average : ℝ :=
  let sixth_grade_minutes : ℝ := 14
  let seventh_grade_minutes : ℝ := 18
  let eighth_grade_minutes : ℝ := 12
  let sixth_to_seventh_ratio : ℝ := 3
  let seventh_to_eighth_ratio : ℝ := 4
  let sports_day_additional_minutes : ℝ := 4
  let days_per_week : ℝ := 7

  let sixth_grade_students : ℝ := seventh_to_eighth_ratio * sixth_to_seventh_ratio
  let seventh_grade_students : ℝ := seventh_to_eighth_ratio
  let eighth_grade_students : ℝ := 1

  let total_students : ℝ := sixth_grade_students + seventh_grade_students + eighth_grade_students

  let average_minutes_with_sports_day : ℝ :=
    (sixth_grade_students * (sixth_grade_minutes * days_per_week + sports_day_additional_minutes) +
     seventh_grade_students * (seventh_grade_minutes * days_per_week + sports_day_additional_minutes) +
     eighth_grade_students * (eighth_grade_minutes * days_per_week + sports_day_additional_minutes)) /
    (total_students * days_per_week)

  15.6

theorem middle_school_running_average_proof : 
  (middle_school_running_average : ℝ) = 15.6 := by sorry

end middle_school_running_average_middle_school_running_average_proof_l1181_118137


namespace expression_value_at_three_l1181_118102

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x ^ (x - 1)) = 30 := by sorry

end expression_value_at_three_l1181_118102


namespace parenthesizations_of_triple_exponent_l1181_118108

/-- Represents the number of distinct parenthesizations of 3^3^3^3 -/
def num_parenthesizations : ℕ := 5

/-- Represents the number of distinct values obtained from different parenthesizations of 3^3^3^3 -/
def num_distinct_values : ℕ := 5

/-- The expression 3^3^3^3 can be parenthesized in 5 different ways, resulting in 5 distinct values -/
theorem parenthesizations_of_triple_exponent :
  num_parenthesizations = num_distinct_values :=
by sorry

#check parenthesizations_of_triple_exponent

end parenthesizations_of_triple_exponent_l1181_118108


namespace quadratic_inequality_solution_set_l1181_118187

/-- If the solution set of ax^2 + bx + c < 0 (a ≠ 0) is R, then a < 0 and b^2 - 4ac < 0 -/
theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) → 
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
by sorry

end quadratic_inequality_solution_set_l1181_118187


namespace condition_relationship_l1181_118174

theorem condition_relationship (p q r s : Prop) 
  (h1 : (r → q) ∧ ¬(q → r))  -- q is necessary but not sufficient for r
  (h2 : (s ↔ r))             -- s is sufficient and necessary for r
  : (s → q) ∧ ¬(q → s) :=    -- s is sufficient but not necessary for q
by sorry

end condition_relationship_l1181_118174


namespace triangle_side_length_l1181_118166

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- arithmetic sequence
  (B = π / 6) →  -- 30° in radians
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- area of triangle
  -- Conclusion
  b = Real.sqrt 3 + 1 := by sorry

end triangle_side_length_l1181_118166


namespace sequence_range_theorem_l1181_118145

def sequence_sum (n : ℕ) : ℚ := (-1)^(n+1) * (1 / 2^n)

def sequence_term (n : ℕ) : ℚ := sequence_sum n - sequence_sum (n-1)

theorem sequence_range_theorem (p : ℚ) : 
  (∃ n : ℕ, (p - sequence_term n) * (p - sequence_term (n+1)) < 0) ↔ 
  (-3/4 < p ∧ p < 1/2) :=
sorry

end sequence_range_theorem_l1181_118145


namespace round_trip_average_speed_l1181_118180

/-- Proves that the average speed of a round trip is 34 mph, given that:
    1. The speed from A to B is 51 mph
    2. The return trip from B to A takes twice as long -/
theorem round_trip_average_speed : ∀ (distance : ℝ) (time : ℝ),
  distance > 0 → time > 0 →
  distance = 51 * time →
  (2 * distance) / (3 * time) = 34 :=
by sorry

end round_trip_average_speed_l1181_118180


namespace servant_payment_theorem_l1181_118193

/-- Calculates the money received by a servant who worked for 9 months, given a yearly salary and uniform value -/
def servant_payment (yearly_salary : ℚ) (uniform_value : ℚ) : ℚ :=
  (yearly_salary * 9 / 12) - uniform_value

/-- The servant payment theorem -/
theorem servant_payment_theorem (yearly_salary : ℚ) (uniform_value : ℚ) 
  (h1 : yearly_salary = 500)
  (h2 : uniform_value = 300) :
  servant_payment yearly_salary uniform_value = 75.03 := by
  sorry

#eval servant_payment 500 300

end servant_payment_theorem_l1181_118193


namespace triangle_inequality_sum_l1181_118170

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + 2*b*c) / (b^2 + c^2) + (b^2 + 2*a*c) / (c^2 + a^2) + (c^2 + 2*a*b) / (a^2 + b^2) > 3 := by
  sorry

end triangle_inequality_sum_l1181_118170


namespace company_works_four_weeks_per_month_l1181_118131

/-- Represents the company's employee and payroll information -/
structure Company where
  initial_employees : ℕ
  additional_employees : ℕ
  hourly_wage : ℚ
  hours_per_day : ℕ
  days_per_week : ℕ
  total_monthly_pay : ℚ

/-- Calculates the number of weeks worked per month -/
def weeks_per_month (c : Company) : ℚ :=
  let total_employees := c.initial_employees + c.additional_employees
  let daily_pay := c.hourly_wage * c.hours_per_day
  let weekly_pay := daily_pay * c.days_per_week
  let total_weekly_pay := weekly_pay * total_employees
  c.total_monthly_pay / total_weekly_pay

/-- Theorem stating that the company's employees work 4 weeks per month -/
theorem company_works_four_weeks_per_month :
  let c : Company := {
    initial_employees := 500,
    additional_employees := 200,
    hourly_wage := 12,
    hours_per_day := 10,
    days_per_week := 5,
    total_monthly_pay := 1680000
  }
  weeks_per_month c = 4 := by
  sorry


end company_works_four_weeks_per_month_l1181_118131


namespace library_shelving_l1181_118152

theorem library_shelving (jason_books_per_time lexi_books_per_time total_books : ℕ) :
  jason_books_per_time = 6 →
  total_books = 102 →
  total_books % jason_books_per_time = 0 →
  total_books % lexi_books_per_time = 0 →
  total_books / jason_books_per_time = total_books / lexi_books_per_time →
  lexi_books_per_time = 6 := by
  sorry

end library_shelving_l1181_118152


namespace range_of_a_l1181_118151

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 3, |x₀^2 - a*x₀ + 4| ≤ 3*x₀) → 1 ≤ a ∧ a ≤ 8 := by
  sorry

end range_of_a_l1181_118151


namespace quadratic_coefficients_identify_coefficients_l1181_118141

theorem quadratic_coefficients (x : ℝ) : 
  5 * x^2 + 1/2 = 6 * x ↔ 5 * x^2 + (-6) * x + 1/2 = 0 :=
by sorry

theorem identify_coefficients :
  ∃ (a b c : ℝ), (∀ x, a * x^2 + b * x + c = 0 ↔ 5 * x^2 + (-6) * x + 1/2 = 0) ∧
  a = 5 ∧ b = -6 ∧ c = 1/2 :=
by sorry

end quadratic_coefficients_identify_coefficients_l1181_118141


namespace sin_80_sin_40_minus_cos_80_cos_40_l1181_118122

theorem sin_80_sin_40_minus_cos_80_cos_40 : 
  Real.sin (80 * π / 180) * Real.sin (40 * π / 180) - 
  Real.cos (80 * π / 180) * Real.cos (40 * π / 180) = 1/2 := by
  sorry

end sin_80_sin_40_minus_cos_80_cos_40_l1181_118122


namespace twentyfifth_triangular_number_l1181_118119

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentyfifth_triangular_number :
  triangular_number 25 = 325 := by sorry

end twentyfifth_triangular_number_l1181_118119
