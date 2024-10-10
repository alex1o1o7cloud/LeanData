import Mathlib

namespace value_of_x_l621_62199

theorem value_of_x : ∀ (x y z w v : ℕ),
  x = y + 7 →
  y = z + 12 →
  z = w + 25 →
  w = v + 5 →
  v = 90 →
  x = 139 := by
  sorry

end value_of_x_l621_62199


namespace triangle_is_equilateral_l621_62141

/-- A triangle with sides a, b, and c is equilateral if b^2 = ac and 2b = a + c -/
theorem triangle_is_equilateral (a b c : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : 2 * b = a + c) : 
  a = b ∧ b = c := by
  sorry

end triangle_is_equilateral_l621_62141


namespace opposite_number_on_line_l621_62137

theorem opposite_number_on_line (a : ℝ) : (a + (a - 6) = 0) → a = 3 := by
  sorry

end opposite_number_on_line_l621_62137


namespace quadratic_roots_transformation_l621_62116

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 - a*r₁ + b = 0) → 
  (r₂^2 - a*r₂ + b = 0) → 
  ∃ (x : ℝ), x^2 - (a^2 + a - 2*b)*x + (a^3 - a*b) = 0 ∧ 
  (x = r₁^2 + r₂ ∨ x = r₁ + r₂^2) :=
by sorry

end quadratic_roots_transformation_l621_62116


namespace square_field_area_specific_field_area_l621_62148

/-- The area of a square field given diagonal travel time and speed -/
theorem square_field_area (travel_time : Real) (speed : Real) : Real :=
  let diagonal_length : Real := speed * (travel_time / 60)
  let side_length : Real := (diagonal_length * 1000) / Real.sqrt 2
  side_length * side_length

/-- Proof of the specific field area -/
theorem specific_field_area : 
  square_field_area 2 3 = 5000 := by
  sorry

end square_field_area_specific_field_area_l621_62148


namespace parallel_line_slope_l621_62127

/-- The slope of a line parallel to 3x + 6y = 15 is -1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ), (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b ↔ 3 * x + 6 * y = 15) →
  m = -1/2 :=
by sorry

end parallel_line_slope_l621_62127


namespace percentage_of_360_equals_162_l621_62198

theorem percentage_of_360_equals_162 : 
  (162 / 360) * 100 = 45 := by sorry

end percentage_of_360_equals_162_l621_62198


namespace track_width_l621_62183

theorem track_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) 
  (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) 
  (h₃ : r₁ - r₂ = 2 * (r₁ - r₂) / 2) : 
  r₁ - r₂ = 10 := by
  sorry

end track_width_l621_62183


namespace vasya_mistake_l621_62157

-- Define the function to calculate the number of digits used for page numbering
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

-- Theorem statement
theorem vasya_mistake :
  ¬ ∃ (n : ℕ), digits_used n = 301 :=
sorry

end vasya_mistake_l621_62157


namespace red_beans_proposition_l621_62171

-- Define a type for lines in the poem
inductive PoemLine
| A : PoemLine
| B : PoemLine
| C : PoemLine
| D : PoemLine

-- Define what a proposition is
def isProposition (line : PoemLine) : Prop :=
  match line with
  | PoemLine.A => true  -- "Red beans grow in the southern country" is a proposition
  | _ => false          -- Other lines are not propositions for this problem

-- Theorem statement
theorem red_beans_proposition :
  isProposition PoemLine.A :=
by sorry

end red_beans_proposition_l621_62171


namespace rectangle_area_change_l621_62115

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.3 * W) = 624 := by sorry

end rectangle_area_change_l621_62115


namespace max_ab_value_l621_62135

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  (∀ c d : ℝ, (∀ x : ℝ, Real.exp (x + 1) ≥ c * x + d) → a * b ≥ c * d) ∧
  a * b = Real.exp 3 / 2 :=
sorry

end max_ab_value_l621_62135


namespace sum_of_coefficients_l621_62192

/-- Given a function f: ℝ → ℝ satisfying the conditions:
    1) f(x+5) = 4x^3 + 5x^2 + 9x + 6
    2) f(x) = ax^3 + bx^2 + cx + d
    Prove that a + b + c + d = -206 -/
theorem sum_of_coefficients (f : ℝ → ℝ) (a b c d : ℝ) :
  (∀ x, f (x + 5) = 4 * x^3 + 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -206 := by
  sorry

end sum_of_coefficients_l621_62192


namespace agathas_bike_purchase_l621_62134

/-- Agatha's bike purchase problem -/
theorem agathas_bike_purchase (frame_cost seat_handlebar_cost front_wheel_cost remaining_money : ℕ) 
  (h1 : frame_cost = 15)
  (h2 : front_wheel_cost = 25)
  (h3 : remaining_money = 20) :
  frame_cost + front_wheel_cost + remaining_money = 60 := by
  sorry

#check agathas_bike_purchase

end agathas_bike_purchase_l621_62134


namespace distance_to_origin_l621_62184

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : x = 2 + 2 * Real.sqrt 30) 
  (h3 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end distance_to_origin_l621_62184


namespace newspaper_pages_read_l621_62102

theorem newspaper_pages_read (jairus_pages arniel_pages total_pages : ℕ) : 
  jairus_pages = 20 →
  arniel_pages = 2 * jairus_pages + 2 →
  total_pages = jairus_pages + arniel_pages →
  total_pages = 62 := by
sorry

end newspaper_pages_read_l621_62102


namespace sqrt_of_three_times_two_five_cubed_l621_62133

theorem sqrt_of_three_times_two_five_cubed (x : ℝ) : 
  x = Real.sqrt (2 * (5^3) + 2 * (5^3) + 2 * (5^3)) → x = 5 * Real.sqrt 30 :=
by
  sorry

end sqrt_of_three_times_two_five_cubed_l621_62133


namespace tangent_circle_properties_l621_62175

/-- A circle with center (1, 2) that is tangent to the x-axis -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem tangent_circle_properties :
  (∀ p ∈ TangentCircle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
  (∃ p ∈ TangentCircle, p.2 = 0) ∧
  (∀ p ∈ TangentCircle, p.2 ≥ 0) :=
sorry

end tangent_circle_properties_l621_62175


namespace doughnuts_per_box_l621_62117

theorem doughnuts_per_box (total : ℕ) (boxes : ℕ) (h1 : total = 48) (h2 : boxes = 4) :
  total / boxes = 12 := by
  sorry

end doughnuts_per_box_l621_62117


namespace extended_quadrilateral_area_l621_62150

-- Define the quadrilateral ABCD
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define the extended quadrilateral A₁B₁C₁D₁
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] extends Quadrilateral V :=
  (A₁ B₁ C₁ D₁ : V)
  (hDA₁ : A₁ - D = 2 • (A - D))
  (hAB₁ : B₁ - A = 2 • (B - A))
  (hBC₁ : C₁ - B = 2 • (C - B))
  (hCD₁ : D₁ - C = 2 • (D - C))

-- Define the area function
noncomputable def area {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : ℝ := sorry

-- State the theorem
theorem extended_quadrilateral_area {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (q : ExtendedQuadrilateral V) :
  area {A := q.A₁, B := q.B₁, C := q.C₁, D := q.D₁} = 5 * area {A := q.A, B := q.B, C := q.C, D := q.D} :=
sorry

end extended_quadrilateral_area_l621_62150


namespace ellipse_condition_l621_62158

-- Define the equation
def equation (x y z m : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m

-- Define what it means for the equation to represent a non-degenerate ellipse when projected onto the xy-plane
def is_nondegenerate_ellipse_projection (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ∃ (z : ℝ), equation x y z m ↔ (x - c)^2 / a + (y - c)^2 / b = 1

-- State the theorem
theorem ellipse_condition (m : ℝ) : 
  is_nondegenerate_ellipse_projection m ↔ m > -21 :=
sorry

end ellipse_condition_l621_62158


namespace integral_evaluation_l621_62104

theorem integral_evaluation : ∫ (x : ℝ) in (0)..(1), (8 / Real.pi) * Real.sqrt (1 - x^2) + 6 * x^2 = 4 := by
  sorry

end integral_evaluation_l621_62104


namespace flea_meeting_configuration_l621_62111

/-- Represents a small triangle on the infinite sheet of triangulated paper -/
structure SmallTriangle where
  x : ℤ
  y : ℤ

/-- Represents the equilateral triangle containing n^2 small triangles -/
def LargeTriangle (n : ℕ) : Set SmallTriangle :=
  { t : SmallTriangle | 0 ≤ t.x ∧ 0 ≤ t.y ∧ t.x + t.y < n }

/-- Represents the set of possible jumps a flea can make -/
def PossibleJumps : List (ℤ × ℤ) := [(1, 0), (-1, 1), (0, -1)]

/-- Defines a valid jump for a flea -/
def ValidJump (t1 t2 : SmallTriangle) : Prop :=
  (t2.x - t1.x, t2.y - t1.y) ∈ PossibleJumps

/-- Theorem: For which positive integers n does there exist an initial configuration
    such that after a finite number of jumps all the n fleas can meet in a single small triangle? -/
theorem flea_meeting_configuration (n : ℕ) :
  (∃ (initial_config : Fin n → SmallTriangle)
     (final_triangle : SmallTriangle)
     (num_jumps : ℕ),
   (∀ i j : Fin n, i ≠ j → initial_config i ≠ initial_config j) ∧
   (∀ i : Fin n, initial_config i ∈ LargeTriangle n) ∧
   (∃ (jump_sequence : Fin n → ℕ → SmallTriangle),
     (∀ i : Fin n, jump_sequence i 0 = initial_config i) ∧
     (∀ i : Fin n, ∀ k : ℕ, k < num_jumps →
       ValidJump (jump_sequence i k) (jump_sequence i (k+1))) ∧
     (∀ i : Fin n, jump_sequence i num_jumps = final_triangle))) ↔
  (n ≥ 1 ∧ n ≠ 2 ∧ n ≠ 4) :=
sorry

end flea_meeting_configuration_l621_62111


namespace remainder_3_101_plus_5_mod_11_l621_62103

theorem remainder_3_101_plus_5_mod_11 : (3^101 + 5) % 11 = 8 := by
  sorry

end remainder_3_101_plus_5_mod_11_l621_62103


namespace duplicate_page_sum_l621_62130

theorem duplicate_page_sum (n : ℕ) (p : ℕ) : 
  p ≤ n →
  n * (n + 1) / 2 + p = 3005 →
  p = 2 :=
sorry

end duplicate_page_sum_l621_62130


namespace opposite_of_negative_sqrt_seven_l621_62136

theorem opposite_of_negative_sqrt_seven (x : ℝ) : 
  x = -Real.sqrt 7 → -x = Real.sqrt 7 :=
by sorry

end opposite_of_negative_sqrt_seven_l621_62136


namespace total_spent_correct_l621_62128

def calculate_total_spent (sandwich_price : Float) (sandwich_discount : Float)
                          (salad_price : Float) (salad_tax : Float)
                          (soda_price : Float) (soda_tax : Float)
                          (tip_percentage : Float) : Float :=
  let discounted_sandwich := sandwich_price * (1 - sandwich_discount)
  let taxed_salad := salad_price * (1 + salad_tax)
  let taxed_soda := soda_price * (1 + soda_tax)
  let subtotal := discounted_sandwich + taxed_salad + taxed_soda
  let total_with_tip := subtotal * (1 + tip_percentage)
  (total_with_tip * 100).round / 100

theorem total_spent_correct :
  calculate_total_spent 10.50 0.15 5.25 0.07 1.75 0.05 0.20 = 19.66 := by
  sorry


end total_spent_correct_l621_62128


namespace cheryl_material_calculation_l621_62189

theorem cheryl_material_calculation (material_used total_bought second_type leftover : ℝ) :
  material_used = 0.21052631578947367 →
  second_type = 2 / 13 →
  leftover = 4 / 26 →
  total_bought = material_used + leftover →
  total_bought = second_type + (0.21052631578947367 : ℝ) :=
by sorry

end cheryl_material_calculation_l621_62189


namespace quadratic_minimum_l621_62145

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 15 ≥ 7) ∧ (∃ x, 2 * x^2 - 8 * x + 15 = 7) := by
  sorry

end quadratic_minimum_l621_62145


namespace regular_tetrahedron_side_edge_length_l621_62164

/-- A regular triangular pyramid (tetrahedron) with specific properties -/
structure RegularTetrahedron where
  /-- The length of the base edge -/
  base_edge : ℝ
  /-- The angle between side faces in radians -/
  side_face_angle : ℝ
  /-- The length of the side edges -/
  side_edge : ℝ
  /-- The base edge is 1 unit long -/
  base_edge_length : base_edge = 1
  /-- The side faces form an angle of 120° (2π/3 radians) with each other -/
  face_angle : side_face_angle = 2 * Real.pi / 3

/-- Theorem stating the length of side edges in a regular tetrahedron with given properties -/
theorem regular_tetrahedron_side_edge_length (t : RegularTetrahedron) : 
  t.side_edge = Real.sqrt 6 / 4 := by
  sorry

end regular_tetrahedron_side_edge_length_l621_62164


namespace inequality_proof_l621_62197

theorem inequality_proof (x : ℝ) : 2/3 < x ∧ x < 5/4 → (4*x - 5)^2 + (3*x - 2)^2 < (x - 3)^2 := by
  sorry

end inequality_proof_l621_62197


namespace triangle_perimeter_l621_62149

theorem triangle_perimeter (a b c : ℝ) : 
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (a^2 + 5*b^2 + c^2 - 4*a*b - 6*b - 10*c + 34 = 0) →
  (a + b + c = 14) := by
sorry

end triangle_perimeter_l621_62149


namespace sin_120_degrees_l621_62193

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l621_62193


namespace arrangement_theorem_l621_62177

/-- The number of ways to arrange 3 people on 6 chairs in a row, 
    such that no two people sit next to each other -/
def arrangement_count : ℕ := 24

/-- The number of chairs in the row -/
def total_chairs : ℕ := 6

/-- The number of people to be seated -/
def people_count : ℕ := 3

/-- Theorem stating that the number of arrangements 
    satisfying the given conditions is 24 -/
theorem arrangement_theorem : 
  arrangement_count = 
    (Nat.factorial people_count) * (total_chairs - people_count - (people_count - 1)) := by
  sorry

end arrangement_theorem_l621_62177


namespace multiplicative_inverse_203_mod_301_l621_62123

theorem multiplicative_inverse_203_mod_301 :
  ∃ x : ℕ, x < 301 ∧ (7236 : ℤ) ≡ x [ZMOD 301] ∧ (203 * x) ≡ 1 [ZMOD 301] := by
  sorry

end multiplicative_inverse_203_mod_301_l621_62123


namespace coordinates_of_P_wrt_origin_l621_62194

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define point P
def P : Point := (-5, 3)

-- Theorem stating that the coordinates of P with respect to the origin are (-5, 3)
theorem coordinates_of_P_wrt_origin :
  P = (-5, 3) := by sorry

end coordinates_of_P_wrt_origin_l621_62194


namespace same_color_prob_is_eleven_thirty_sixths_l621_62167

/-- A die with 12 sides and specific color distribution -/
structure TwelveSidedDie :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (golden : ℕ)
  (total_sides : red + blue + green + golden = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : TwelveSidedDie) : ℚ :=
  (d.red^2 + d.blue^2 + d.green^2 + d.golden^2) / 144

/-- Theorem stating the probability of two 12-sided dice showing the same color -/
theorem same_color_prob_is_eleven_thirty_sixths :
  ∀ d : TwelveSidedDie,
  d.red = 3 → d.blue = 5 → d.green = 3 → d.golden = 1 →
  same_color_probability d = 11 / 36 :=
sorry

end same_color_prob_is_eleven_thirty_sixths_l621_62167


namespace michael_has_two_cats_l621_62182

/-- The number of dogs Michael has -/
def num_dogs : ℕ := 3

/-- The cost per night per animal for pet-sitting -/
def cost_per_animal : ℕ := 13

/-- The total cost for pet-sitting -/
def total_cost : ℕ := 65

/-- The number of cats Michael has -/
def num_cats : ℕ := (total_cost - num_dogs * cost_per_animal) / cost_per_animal

theorem michael_has_two_cats : num_cats = 2 := by
  sorry

end michael_has_two_cats_l621_62182


namespace garrison_problem_l621_62159

/-- Represents the number of men in a garrison and their provisions --/
structure Garrison where
  initialMen : ℕ
  initialDays : ℕ
  reinforcementMen : ℕ
  remainingDays : ℕ
  reinforcementArrivalDay : ℕ

/-- Calculates the initial number of men in the garrison --/
def calculateInitialMen (g : Garrison) : ℕ :=
  (g.initialDays - g.reinforcementArrivalDay) * g.initialMen / 
  (g.initialDays - g.reinforcementArrivalDay - g.remainingDays)

/-- Theorem stating that given the conditions, the initial number of men is 2000 --/
theorem garrison_problem (g : Garrison) 
  (h1 : g.initialDays = 65)
  (h2 : g.reinforcementMen = 3000)
  (h3 : g.remainingDays = 20)
  (h4 : g.reinforcementArrivalDay = 15) :
  calculateInitialMen g = 2000 := by
  sorry

#eval calculateInitialMen { initialMen := 2000, initialDays := 65, reinforcementMen := 3000, remainingDays := 20, reinforcementArrivalDay := 15 }

end garrison_problem_l621_62159


namespace rational_solution_cosine_equation_l621_62187

theorem rational_solution_cosine_equation (q : ℚ) 
  (h1 : 0 < q) (h2 : q < 1) 
  (h3 : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2/3 := by
sorry

end rational_solution_cosine_equation_l621_62187


namespace F_is_odd_l621_62126

-- Define the function f on the real numbers
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (-x)

-- Theorem statement
theorem F_is_odd (f : ℝ → ℝ) : 
  ∀ x : ℝ, F f x = -(F f (-x)) := by
  sorry

end F_is_odd_l621_62126


namespace quadratic_zero_point_range_l621_62151

/-- The quadratic function f(x) = x^2 - 2x + a has a zero point in the interval (-1,3) 
    if and only if a is in the range (-3,1]. -/
theorem quadratic_zero_point_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1) 3 ∧ x^2 - 2*x + a = 0) ↔ a ∈ Set.Ioc (-3) 1 := by
  sorry

end quadratic_zero_point_range_l621_62151


namespace arithmetic_sequence_100th_term_l621_62195

/-- Given an arithmetic sequence {a_n} with a_1 = 1 and common difference d = 3,
    prove that the 100th term is equal to 298. -/
theorem arithmetic_sequence_100th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 1) → 
    (∀ n : ℕ, a (n + 1) - a n = 3) → 
    (a 100 = 298) := by
  sorry

end arithmetic_sequence_100th_term_l621_62195


namespace cos_shift_symmetry_axis_l621_62146

/-- The axis of symmetry for a cosine function shifted left by π/12 -/
theorem cos_shift_symmetry_axis (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * (x + π / 12))
  ∀ x : ℝ, f (k * π / 2 - π / 12 - x) = f (k * π / 2 - π / 12 + x) := by
  sorry

end cos_shift_symmetry_axis_l621_62146


namespace problem_solution_l621_62172

theorem problem_solution : ∃! x : ℝ, 0.8 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.6 := by
  sorry

end problem_solution_l621_62172


namespace min_value_theorem_l621_62179

theorem min_value_theorem (a : ℝ) :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x > 0 → y = -x^2 + 3 * Real.log x →
  (a - x)^2 + (a + 2 - y)^2 ≥ min :=
by sorry

end min_value_theorem_l621_62179


namespace inequality_holds_l621_62110

theorem inequality_holds (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : b < 3) :
  b * 2^a < a * 2^b := by
  sorry

end inequality_holds_l621_62110


namespace sin_135_degrees_l621_62173

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l621_62173


namespace younger_person_age_l621_62170

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's current age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end younger_person_age_l621_62170


namespace max_value_of_ab_l621_62156

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ab ≤ 1/8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ a₀ * b₀ = 1/8 :=
sorry

end max_value_of_ab_l621_62156


namespace floor_equation_solution_l621_62186

theorem floor_equation_solution (A B : ℝ) (hA : A ≥ 0) (hB : B ≥ 0) :
  (∀ x : ℝ, x > 1 → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  A = 0 ∧ B = 1 := by
sorry

end floor_equation_solution_l621_62186


namespace integral_sin_cubed_over_cos_fifth_l621_62185

theorem integral_sin_cubed_over_cos_fifth (x : Real) :
  let f := fun (x : Real) => (1 / (4 * (Real.cos x)^4)) - (1 / (2 * (Real.cos x)^2))
  deriv f x = (Real.sin x)^3 / (Real.cos x)^5 := by
  sorry

end integral_sin_cubed_over_cos_fifth_l621_62185


namespace mrsHiltFramePerimeter_l621_62142

/-- An irregular octagon with specified side lengths -/
structure IrregularOctagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  side7 : ℝ
  side8 : ℝ

/-- Calculate the perimeter of an irregular octagon -/
def perimeter (o : IrregularOctagon) : ℝ :=
  o.side1 + o.side2 + o.side3 + o.side4 + o.side5 + o.side6 + o.side7 + o.side8

/-- Mrs. Hilt's irregular octagonal picture frame -/
def mrsHiltFrame : IrregularOctagon :=
  { side1 := 10
    side2 := 9
    side3 := 11
    side4 := 6
    side5 := 7
    side6 := 2
    side7 := 3
    side8 := 4 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular octagonal picture frame is 52 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 52 := by
  sorry

end mrsHiltFramePerimeter_l621_62142


namespace unique_solution_exists_l621_62181

/-- Represents a digit from 0 to 7 -/
def Digit := Fin 8

/-- Converts a three-digit number to its integer representation -/
def toInt (a b c : Digit) : Nat := a.val * 100 + b.val * 10 + c.val

/-- Converts a two-digit number to its integer representation -/
def toInt2 (d e : Digit) : Nat := d.val * 10 + e.val

theorem unique_solution_exists (a b c d e f g h : Digit) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h)
  (abc_eq : toInt a b c = 146)
  (equation : toInt a b c + toInt2 d e = toInt f g h) :
  toInt2 d e = 57 := by
  sorry

end unique_solution_exists_l621_62181


namespace intersection_triangle_is_right_angle_l621_62176

/-- An ellipse with semi-major axis √m and semi-minor axis 1 -/
structure Ellipse (m : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / m + y^2 = 1)
  (m_gt_one : m > 1)

/-- A hyperbola with semi-major axis √n and semi-minor axis 1 -/
structure Hyperbola (n : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / n - y^2 = 1)
  (n_pos : n > 0)

/-- The foci of a conic section -/
structure Foci :=
  (F₁ F₂ : ℝ × ℝ)

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Theorem: The triangle formed by the foci and an intersection point of an ellipse and hyperbola with the same foci is a right triangle -/
theorem intersection_triangle_is_right_angle
  (m n : ℝ)
  (E : Ellipse m)
  (H : Hyperbola n)
  (F : Foci)
  (P : Point)
  (h₁ : E.x = P.1 ∧ E.y = P.2)  -- P is on the ellipse
  (h₂ : H.x = P.1 ∧ H.y = P.2)  -- P is on the hyperbola
  (h₃ : F.F₁ ≠ F.F₂)  -- The foci are distinct
  : ∃ (A B C : ℝ),
    (P.1 - F.F₁.1)^2 + (P.2 - F.F₁.2)^2 = A^2 ∧
    (P.1 - F.F₂.1)^2 + (P.2 - F.F₂.2)^2 = B^2 ∧
    (F.F₁.1 - F.F₂.1)^2 + (F.F₁.2 - F.F₂.2)^2 = C^2 ∧
    A^2 + B^2 = C^2 :=
  sorry

end intersection_triangle_is_right_angle_l621_62176


namespace only_statements_1_and_2_correct_l621_62125

-- Define the structure of a programming statement
inductive ProgrammingStatement
| Input : String → ProgrammingStatement
| Output : String → ProgrammingStatement
| Assignment : String → String → ProgrammingStatement

-- Define the property of being a correct statement
def is_correct (s : ProgrammingStatement) : Prop :=
  match s with
  | ProgrammingStatement.Input _ => true
  | ProgrammingStatement.Output _ => false
  | ProgrammingStatement.Assignment lhs rhs => lhs ≠ rhs

-- Define the four statements from the problem
def statement1 : ProgrammingStatement := ProgrammingStatement.Input "x=3"
def statement2 : ProgrammingStatement := ProgrammingStatement.Input "A, B, C"
def statement3 : ProgrammingStatement := ProgrammingStatement.Output "A+B=C"
def statement4 : ProgrammingStatement := ProgrammingStatement.Assignment "3" "A"

-- Theorem to prove
theorem only_statements_1_and_2_correct :
  is_correct statement1 ∧ 
  is_correct statement2 ∧ 
  ¬is_correct statement3 ∧ 
  ¬is_correct statement4 :=
sorry

end only_statements_1_and_2_correct_l621_62125


namespace computer_literate_female_employees_l621_62161

theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ) 
  (male_literate_percentage : ℚ) 
  (total_literate_percentage : ℚ) 
  (h1 : total_employees = 1400)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_literate_percentage - 
  (↑total_employees * (1 - female_percentage) * male_literate_percentage) = 588 := by
  sorry

#check computer_literate_female_employees

end computer_literate_female_employees_l621_62161


namespace root_properties_l621_62132

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^20 - 123*x^10 + 1

-- Define the polynomial g
def g (x : ℝ) : ℝ := x^4 + 3*x^3 + 4*x^2 + 2*x + 1

theorem root_properties (a β : ℝ) : 
  (f a = 0 → f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0) ∧
  (g β = 0 → g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0) :=
by sorry

end root_properties_l621_62132


namespace yoga_studio_total_people_l621_62105

theorem yoga_studio_total_people :
  let num_men : ℕ := 8
  let num_women : ℕ := 6
  let avg_weight_men : ℝ := 190
  let avg_weight_women : ℝ := 120
  let avg_weight_all : ℝ := 160
  num_men + num_women = 14 := by
  sorry

end yoga_studio_total_people_l621_62105


namespace seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l621_62154

def is_beautiful (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + b^2 ∨ n = a^2 - b^2

theorem seven_consecutive_beautiful_numbers (k : ℤ) (hk : k ≥ 0) :
  ∃ n : ℤ, n ≥ k ∧ 
    (∀ i : ℤ, 0 ≤ i ∧ i < 7 → is_beautiful (8*n + i - 1)) ∧
    ¬(∀ i : ℤ, 0 ≤ i ∧ i < 8 → is_beautiful (8*n + i - 1)) :=
sorry

theorem odd_numbers_beautiful (n : ℤ) :
  n % 2 = 1 → is_beautiful n :=
sorry

theorem divisible_by_four_beautiful (n : ℤ) :
  n % 4 = 0 → is_beautiful n :=
sorry

theorem not_beautiful_mod_eight_six (n : ℤ) :
  n % 8 = 6 → ¬is_beautiful n :=
sorry

end seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l621_62154


namespace min_value_m_l621_62178

theorem min_value_m (m : ℝ) (h1 : m > 0)
  (h2 : ∀ x : ℝ, x > 1 → 2 * Real.exp (2 * m * x) - Real.log x / m ≥ 0) :
  m ≥ 1 / (2 * Real.exp 1) :=
sorry

end min_value_m_l621_62178


namespace a_equals_3_necessary_not_sufficient_l621_62118

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax-2y-1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -2, c := -1 }

/-- The second line 6x-4y+c=0 -/
def line2 (c : ℝ) : Line :=
  { a := 6, b := -4, c := c }

theorem a_equals_3_necessary_not_sufficient :
  (∀ c, parallel (line1 3) (line2 c)) ∧
  (∃ a c, a ≠ 3 ∧ parallel (line1 a) (line2 c)) :=
sorry

end a_equals_3_necessary_not_sufficient_l621_62118


namespace all_equilateral_triangles_similar_l621_62153

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Similarity relation between two equilateral triangles -/
def similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

/-- All angles in an equilateral triangle are 60° -/
axiom equilateral_angle (t : EquilateralTriangle) : 
  ∀ angle, angle = 60

/-- Theorem: Any two equilateral triangles are similar -/
theorem all_equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 := by
  sorry

end all_equilateral_triangles_similar_l621_62153


namespace not_all_altitudes_inside_l621_62140

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define an altitude of a triangle
def altitude (t : Triangle) (v : Fin 3) : Set (ℝ × ℝ) :=
  sorry

-- Define the property of being inside a triangle
def inside_triangle (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Define different types of triangles
def is_acute_triangle (t : Triangle) : Prop :=
  sorry

def is_right_triangle (t : Triangle) : Prop :=
  sorry

def is_obtuse_triangle (t : Triangle) : Prop :=
  sorry

-- The theorem to be proven
theorem not_all_altitudes_inside : ¬ ∀ (t : Triangle), 
  (∀ (v : Fin 3), ∀ (p : ℝ × ℝ), p ∈ altitude t v → inside_triangle t p) :=
sorry

end not_all_altitudes_inside_l621_62140


namespace average_after_removing_two_numbers_l621_62163

/-- Given a list of 50 numbers with an average of 62, prove that if we remove 45 and 55 from the list,
    the average of the remaining numbers is 62.5 -/
theorem average_after_removing_two_numbers
  (numbers : List ℝ)
  (h_count : numbers.length = 50)
  (h_avg : numbers.sum / numbers.length = 62)
  (h_contains_45 : 45 ∈ numbers)
  (h_contains_55 : 55 ∈ numbers) :
  let remaining := numbers.filter (λ x => x ≠ 45 ∧ x ≠ 55)
  remaining.sum / remaining.length = 62.5 := by
sorry


end average_after_removing_two_numbers_l621_62163


namespace inequality_solution_set_l621_62119

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end inequality_solution_set_l621_62119


namespace min_value_of_arithmetic_sequence_l621_62169

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem min_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_eq : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ q : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end min_value_of_arithmetic_sequence_l621_62169


namespace trigonometric_inequality_l621_62124

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (h2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ ≤ Real.pi / 2)
  (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
      (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
      (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ∧
  (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
  (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
  (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ≤ 
  (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) := by
  sorry

end trigonometric_inequality_l621_62124


namespace brick_height_is_6cm_l621_62122

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions := ⟨800, 600, 22.5⟩

/-- The known dimensions of a brick in centimeters -/
def brick_dimensions (height : ℝ) : Dimensions := ⟨80, 11.25, height⟩

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧ 
  volume wall_dimensions = ↑num_bricks * volume (brick_dimensions h) := by
  sorry

end brick_height_is_6cm_l621_62122


namespace large_rectangle_area_l621_62190

def small_rectangle_perimeter : ℕ := 20

def large_rectangle_side_difference : ℕ := 2

def valid_areas : Set ℕ := {3300, 4000, 4500, 4800, 4900}

theorem large_rectangle_area (l w : ℕ) :
  (l + w = small_rectangle_perimeter / 2) →
  (l > 0 ∧ w > 0) →
  ((l + large_rectangle_side_difference) * (w + large_rectangle_side_difference) * 100) ∈ valid_areas :=
by sorry

end large_rectangle_area_l621_62190


namespace can_form_123_l621_62100

-- Define a data type for arithmetic expressions
inductive Expr
  | num : Nat → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

-- Define a function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

-- Define a predicate to check if an expression uses all numbers exactly once
def usesAllNumbers (e : Expr) : Prop := sorry

-- Theorem stating that 123 can be formed
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end can_form_123_l621_62100


namespace kyungsoo_string_shorter_l621_62196

/-- Conversion factor from centimeters to millimeters -/
def cm_to_mm : ℚ := 10

/-- Length of Inhyuk's string in centimeters -/
def inhyuk_length_cm : ℚ := 97.5

/-- Base length of Kyungsoo's string in centimeters -/
def kyungsoo_base_length_cm : ℚ := 97

/-- Additional length of Kyungsoo's string in millimeters -/
def kyungsoo_additional_length_mm : ℚ := 3

/-- Theorem stating that Kyungsoo's string is shorter than Inhyuk's -/
theorem kyungsoo_string_shorter :
  kyungsoo_base_length_cm * cm_to_mm + kyungsoo_additional_length_mm <
  inhyuk_length_cm * cm_to_mm := by
  sorry

end kyungsoo_string_shorter_l621_62196


namespace chord_length_of_concentric_circles_l621_62112

/-- Given two concentric circles with radii R and r, where the area of the annulus
    between them is 12½π square inches, the length of the chord of the larger circle
    which is tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_of_concentric_circles (R r : ℝ) :
  R > r →
  π * R^2 - π * r^2 = 25 / 2 * π →
  ∃ (c : ℝ), c^2 = R^2 - r^2 ∧ c = 5 * Real.sqrt 2 :=
by sorry

end chord_length_of_concentric_circles_l621_62112


namespace no_intersection_l621_62160

-- Define the two functions
def f (x : ℝ) : ℝ := |2 * x + 5|
def g (x : ℝ) : ℝ := -|3 * x - 2|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end no_intersection_l621_62160


namespace lunch_cost_per_person_l621_62131

theorem lunch_cost_per_person (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) : 
  total_price = 207 ∧ num_people = 15 ∧ gratuity_rate = 0.15 →
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
sorry

end lunch_cost_per_person_l621_62131


namespace simplify_algebraic_expression_l621_62121

theorem simplify_algebraic_expression (a : ℝ) : 2*a - 7*a + 4*a = -a := by
  sorry

end simplify_algebraic_expression_l621_62121


namespace extreme_value_theorem_l621_62129

theorem extreme_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 * a * b → 4 * x + 3 * y ≥ 4 * a + 3 * b ∧ 4 * x + 3 * y ≤ 3 :=
by sorry

end extreme_value_theorem_l621_62129


namespace absolute_value_equation_solution_l621_62120

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 10| + |x - 14| = |2*x - 24| :=
by
  sorry

end absolute_value_equation_solution_l621_62120


namespace stratified_sampling_l621_62188

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 300

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 90

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 150

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := 60

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Theorem stating the correct stratified sampling for each teacher category -/
theorem stratified_sampling :
  (senior_teachers * sample_size) / total_teachers = 18 ∧
  (intermediate_teachers * sample_size) / total_teachers = 30 ∧
  (junior_teachers * sample_size) / total_teachers = 12 := by
  sorry

end stratified_sampling_l621_62188


namespace derivative_one_implies_x_is_one_l621_62147

open Real

theorem derivative_one_implies_x_is_one (f : ℝ → ℝ) (x₀ : ℝ) :
  (f = λ x => x * log x) →
  (deriv f x₀ = 1) →
  x₀ = 1 := by
sorry

end derivative_one_implies_x_is_one_l621_62147


namespace tetrahedron_projection_ratio_l621_62165

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_greatest : a ≥ max b (max c (max d (max e f)))

/-- The ratio of projection areas for a tetrahedron -/
noncomputable def projection_area_ratio (t : Tetrahedron) : ℝ := sorry

/-- Theorem: For every tetrahedron, there exist two planes such that 
    the ratio of projection areas on those planes is not less than √2 -/
theorem tetrahedron_projection_ratio (t : Tetrahedron) : 
  projection_area_ratio t ≥ Real.sqrt 2 := by sorry

end tetrahedron_projection_ratio_l621_62165


namespace cylinder_height_l621_62166

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylinder_height (d V : ℝ) (h_d : d = 4) (h_V : V = 20) :
  let r := d / 2
  let h := V / (π * r^2)
  h = 5 / π :=
by sorry

end cylinder_height_l621_62166


namespace divisor_problem_l621_62106

theorem divisor_problem (d : ℕ) : d > 0 ∧ 109 = 9 * d + 1 → d = 12 := by
  sorry

end divisor_problem_l621_62106


namespace kindergarten_cats_count_l621_62144

/-- Represents the number of children in each category in the kindergarten. -/
structure KindergartenPets where
  total : ℕ
  dogsOnly : ℕ
  bothPets : ℕ
  catsOnly : ℕ

/-- Calculates the total number of children with cats in the kindergarten. -/
def childrenWithCats (k : KindergartenPets) : ℕ :=
  k.catsOnly + k.bothPets

/-- Theorem stating the number of children with cats in the kindergarten. -/
theorem kindergarten_cats_count (k : KindergartenPets)
    (h1 : k.total = 30)
    (h2 : k.dogsOnly = 18)
    (h3 : k.bothPets = 6)
    (h4 : k.total = k.dogsOnly + k.catsOnly + k.bothPets) :
    childrenWithCats k = 12 := by
  sorry

#check kindergarten_cats_count

end kindergarten_cats_count_l621_62144


namespace symmetric_point_xoy_plane_l621_62162

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xoy plane --/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetric_point_xoy_plane :
  let M : Point3D := { x := 2, y := 5, z := 8 }
  let N : Point3D := symmetricXOY M
  N = { x := 2, y := 5, z := -8 } := by
  sorry

end symmetric_point_xoy_plane_l621_62162


namespace cos_two_beta_l621_62107

theorem cos_two_beta (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 7) (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  Real.cos (2 * β) = -3/5 := by
  sorry

end cos_two_beta_l621_62107


namespace correct_percentage_l621_62114

theorem correct_percentage (y : ℕ) (y_pos : y > 0) : 
  let total := 7 * y
  let incorrect := 2 * y
  let correct := total - incorrect
  (correct : ℚ) / total * 100 = 500 / 7 := by
  sorry

end correct_percentage_l621_62114


namespace monotonic_sequence_bound_l621_62113

theorem monotonic_sequence_bound (b : ℝ) :
  (∀ n : ℕ, (n + 1)^2 + b*(n + 1) > n^2 + b*n) →
  b > -3 := by
sorry

end monotonic_sequence_bound_l621_62113


namespace q_div_p_eq_275_l621_62152

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_cards cards_drawn

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (3300 : ℚ) / Nat.choose total_cards cards_drawn

/-- The main theorem stating that q/p = 275 -/
theorem q_div_p_eq_275 : q / p = 275 := by sorry

end q_div_p_eq_275_l621_62152


namespace cylinder_volume_l621_62191

/-- The volume of a cylinder with base radius 1 cm and generatrix length 2 cm is 2π cm³ -/
theorem cylinder_volume (π : ℝ) : ℝ := by
  sorry

#check cylinder_volume

end cylinder_volume_l621_62191


namespace max_a_is_correct_l621_62180

/-- The quadratic function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The maximum value of a for which f(x) is increasing when x ≤ a -/
def max_a : ℝ := 1

theorem max_a_is_correct :
  ∀ a : ℝ, (∀ x y : ℝ, x ≤ y → y ≤ a → f x ≤ f y) → a ≤ max_a :=
by sorry

end max_a_is_correct_l621_62180


namespace fish_pond_population_l621_62101

/-- Represents the total number of fish in a pond using the mark and recapture method. -/
def totalFishInPond (initialMarked : ℕ) (secondCatch : ℕ) (markedInSecondCatch : ℕ) : ℕ :=
  (initialMarked * secondCatch) / markedInSecondCatch

/-- Theorem stating that under the given conditions, the total number of fish in the pond is 2400. -/
theorem fish_pond_population :
  let initialMarked : ℕ := 80
  let secondCatch : ℕ := 150
  let markedInSecondCatch : ℕ := 5
  totalFishInPond initialMarked secondCatch markedInSecondCatch = 2400 :=
by sorry


end fish_pond_population_l621_62101


namespace fraction_simplification_l621_62139

theorem fraction_simplification (a b c d : ℕ) (h1 : a = 2637) (h2 : b = 18459) (h3 : c = 5274) (h4 : d = 36918) :
  a / b = 1 / 7 → c / d = 1 / 7 := by
  sorry

end fraction_simplification_l621_62139


namespace quadratic_equation_properties_l621_62174

theorem quadratic_equation_properties (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 9 * x₁ + c = 0 ∧ 3 * x₂^2 - 9 * x₂ + c = 0) →
  (c < 6.75 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by sorry

end quadratic_equation_properties_l621_62174


namespace teena_yoe_distance_l621_62155

/-- Calculates the initial distance between two drivers given their speeds and future relative position --/
def initialDistance (teenaSpeed yoeSpeed : ℝ) (timeAhead : ℝ) (distanceAhead : ℝ) : ℝ :=
  (teenaSpeed - yoeSpeed) * timeAhead - distanceAhead

theorem teena_yoe_distance :
  let teenaSpeed : ℝ := 55
  let yoeSpeed : ℝ := 40
  let timeAhead : ℝ := 1.5  -- 90 minutes in hours
  let distanceAhead : ℝ := 15
  initialDistance teenaSpeed yoeSpeed timeAhead distanceAhead = 7.5 := by
  sorry

#eval initialDistance 55 40 1.5 15

end teena_yoe_distance_l621_62155


namespace simplify_expression1_simplify_expression2_l621_62108

-- Problem 1
theorem simplify_expression1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by sorry

end simplify_expression1_simplify_expression2_l621_62108


namespace tracy_dogs_food_consumption_l621_62143

/-- Proves that Tracy's two dogs consume 4 pounds of food per day -/
theorem tracy_dogs_food_consumption :
  let num_dogs : ℕ := 2
  let cups_per_meal_per_dog : ℚ := 3/2
  let meals_per_day : ℕ := 3
  let cups_per_pound : ℚ := 9/4
  
  let total_cups_per_day : ℚ := num_dogs * cups_per_meal_per_dog * meals_per_day
  let total_pounds_per_day : ℚ := total_cups_per_day / cups_per_pound
  
  total_pounds_per_day = 4 := by sorry

end tracy_dogs_food_consumption_l621_62143


namespace dinner_cost_difference_l621_62168

theorem dinner_cost_difference (initial_amount : ℝ) (first_course_cost : ℝ) (remaining_amount : ℝ) : 
  initial_amount = 60 →
  first_course_cost = 15 →
  remaining_amount = 20 →
  ∃ (second_course_cost : ℝ),
    initial_amount = first_course_cost + second_course_cost + (0.25 * second_course_cost) + remaining_amount ∧
    second_course_cost - first_course_cost = 5 :=
by
  sorry

end dinner_cost_difference_l621_62168


namespace slope_is_negative_one_l621_62138

/-- The slope of a line through two points is -1 -/
theorem slope_is_negative_one (P Q : ℝ × ℝ) : 
  P = (-3, 8) → 
  Q.1 = 5 → 
  Q.2 = 0 → 
  (Q.2 - P.2) / (Q.1 - P.1) = -1 := by
sorry

end slope_is_negative_one_l621_62138


namespace complex_expression_evaluation_l621_62109

theorem complex_expression_evaluation :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I :=
by sorry

end complex_expression_evaluation_l621_62109
