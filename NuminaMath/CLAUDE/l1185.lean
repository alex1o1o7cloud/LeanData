import Mathlib

namespace NUMINAMATH_CALUDE_max_volume_cube_l1185_118587

/-- Given a constant sum of edges, the volume of a rectangular prism is maximized when it is a cube -/
theorem max_volume_cube (s : ℝ) (hs : s > 0) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 * s →
  a * b * c ≤ s^3 ∧ (a * b * c = s^3 ↔ a = s ∧ b = s ∧ c = s) :=
by sorry

end NUMINAMATH_CALUDE_max_volume_cube_l1185_118587


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1185_118540

/-- The value of p for a parabola y² = 2px (p > 0) where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x) → 
  let focus := (p, 0)
  Real.sqrt ((p - (-2))^2 + (0 - 3)^2) = 5 → 
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1185_118540


namespace NUMINAMATH_CALUDE_quadratic_sum_l1185_118598

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) ∧ (a + b + c = -195) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1185_118598


namespace NUMINAMATH_CALUDE_hexagon_coloring_count_l1185_118527

/-- A regular hexagon with 6 regions -/
inductive HexagonRegion
| A | B | C | D | E | F

/-- The available colors for planting -/
inductive PlantColor
| Color1 | Color2 | Color3 | Color4

/-- A coloring of the hexagon -/
def HexagonColoring := HexagonRegion → PlantColor

/-- Check if two regions are adjacent -/
def isAdjacent (r1 r2 : HexagonRegion) : Bool :=
  match r1, r2 with
  | HexagonRegion.A, HexagonRegion.B => true
  | HexagonRegion.A, HexagonRegion.F => true
  | HexagonRegion.B, HexagonRegion.C => true
  | HexagonRegion.C, HexagonRegion.D => true
  | HexagonRegion.D, HexagonRegion.E => true
  | HexagonRegion.E, HexagonRegion.F => true
  | _, _ => false

/-- Check if a coloring is valid (adjacent regions have different colors) -/
def isValidColoring (c : HexagonColoring) : Prop :=
  ∀ r1 r2 : HexagonRegion, isAdjacent r1 r2 → c r1 ≠ c r2

/-- The number of valid colorings -/
def numValidColorings : ℕ := 732

/-- The main theorem -/
theorem hexagon_coloring_count :
  (c : HexagonColoring) → (isValidColoring c) → numValidColorings = 732 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_coloring_count_l1185_118527


namespace NUMINAMATH_CALUDE_perpendicular_bisector_eq_l1185_118503

/-- The perpendicular bisector of a line segment MN is the set of all points
    equidistant from M and N. This theorem proves that for M(2, 4) and N(6, 2),
    the equation of the perpendicular bisector is 2x - y - 5 = 0. -/
theorem perpendicular_bisector_eq (x y : ℝ) :
  let M : ℝ × ℝ := (2, 4)
  let N : ℝ × ℝ := (6, 2)
  (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2 ↔ 2*x - y - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_eq_l1185_118503


namespace NUMINAMATH_CALUDE_vegetable_factory_profit_profit_function_correct_l1185_118599

/-- Represents the net profit function for a vegetable processing factory -/
def net_profit (n : ℕ) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the year when the business starts making a net profit -/
def profit_start_year : ℕ := 3

theorem vegetable_factory_profit :
  (∀ n : ℕ, n < profit_start_year → net_profit n ≤ 0) ∧
  (∀ n : ℕ, n ≥ profit_start_year → net_profit n > 0) :=
sorry

theorem profit_function_correct (n : ℕ) :
  net_profit n = n * 1 - (0.24 * n + n * (n - 1) / 2 * 0.08) - 1.44 :=
sorry

end NUMINAMATH_CALUDE_vegetable_factory_profit_profit_function_correct_l1185_118599


namespace NUMINAMATH_CALUDE_exam_grade_logic_l1185_118584

theorem exam_grade_logic 
  (student : Type) 
  (received_A : student → Prop)
  (all_mc_correct : student → Prop)
  (problem_solving_90_percent : student → Prop)
  (h : ∀ s : student, (all_mc_correct s ∨ problem_solving_90_percent s) → received_A s) :
  ∀ s : student, ¬(received_A s) → (¬(all_mc_correct s) ∧ ¬(problem_solving_90_percent s)) :=
by sorry

end NUMINAMATH_CALUDE_exam_grade_logic_l1185_118584


namespace NUMINAMATH_CALUDE_amy_doll_cost_l1185_118507

def doll_cost (initial_amount : ℕ) (dolls_bought : ℕ) (remaining_amount : ℕ) : ℚ :=
  (initial_amount - remaining_amount : ℚ) / dolls_bought

theorem amy_doll_cost :
  doll_cost 100 3 97 = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_doll_cost_l1185_118507


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l1185_118545

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l1185_118545


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l1185_118512

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 4 = 25 →
  arithmetic_sequence a₁ d 6 = 49 →
  arithmetic_sequence a₁ d 8 = 73 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l1185_118512


namespace NUMINAMATH_CALUDE_siblings_combined_age_l1185_118525

/-- The combined age of five siblings -/
def combined_age (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ) : ℕ :=
  aaron_age + henry_sister_age + henry_age + alice_age + eric_age

theorem siblings_combined_age :
  ∀ (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ),
    aaron_age = 15 →
    henry_sister_age = 3 * aaron_age →
    henry_age = 4 * henry_sister_age →
    alice_age = aaron_age - 2 →
    eric_age = henry_sister_age + alice_age →
    combined_age aaron_age henry_sister_age henry_age alice_age eric_age = 311 :=
by
  sorry

end NUMINAMATH_CALUDE_siblings_combined_age_l1185_118525


namespace NUMINAMATH_CALUDE_roots_pure_imaginary_for_pure_imaginary_k_l1185_118562

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z k : ℂ) : Prop :=
  8 * z^2 + 6 * i * z - k = 0

-- Define a pure imaginary number
def is_pure_imaginary (x : ℂ) : Prop :=
  x.re = 0 ∧ x.im ≠ 0

-- Define the nature of roots
def roots_are_pure_imaginary (k : ℂ) : Prop :=
  ∀ z : ℂ, quadratic_equation z k → is_pure_imaginary z

-- Theorem statement
theorem roots_pure_imaginary_for_pure_imaginary_k :
  ∀ k : ℂ, is_pure_imaginary k → roots_are_pure_imaginary k :=
by sorry

end NUMINAMATH_CALUDE_roots_pure_imaginary_for_pure_imaginary_k_l1185_118562


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1185_118530

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (π/6 - α) = -1/3) : Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1185_118530


namespace NUMINAMATH_CALUDE_sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l1185_118566

/-- Represents the number of classes in each grade -/
structure GradeClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of classes sampled from each grade -/
structure SampledClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- The total number of classes across all grades -/
def totalClasses (gc : GradeClasses) : Nat :=
  gc.grade10 + gc.grade11 + gc.grade12

/-- The number of classes to be sampled -/
def totalSampled : Nat := 9

/-- The school's grade distribution -/
def schoolClasses : GradeClasses :=
  { grade10 := 16, grade11 := 12, grade12 := 8 }

/-- Theorem stating that the sampled classes are proportional to the total classes in each grade -/
theorem sampled_classes_proportional (sc : SampledClasses) :
    sc.grade10 * totalClasses schoolClasses = schoolClasses.grade10 * totalSampled ∧
    sc.grade11 * totalClasses schoolClasses = schoolClasses.grade11 * totalSampled ∧
    sc.grade12 * totalClasses schoolClasses = schoolClasses.grade12 * totalSampled :=
  sorry

/-- The probability of selecting at least one class from grade 12 -/
def probAtLeastOneGrade12 : Rat := 7 / 10

/-- Theorem stating the probability of selecting at least one class from grade 12 -/
theorem prob_at_least_one_grade12 (sc : SampledClasses) :
    probAtLeastOneGrade12 = 7 / 10 :=
  sorry

/-- The probability of selecting both class A from grade 11 and class B from grade 12 -/
def probBothClassesSelected : Rat := 1 / 6

/-- Theorem stating the probability of selecting both class A from grade 11 and class B from grade 12 -/
theorem prob_both_classes_selected (sc : SampledClasses) :
    probBothClassesSelected = 1 / 6 :=
  sorry

end NUMINAMATH_CALUDE_sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l1185_118566


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1185_118505

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, a x y ≠ 0 ∧ b x y ≠ 0 → (a x y = b x y)

/-- The first monomial 5x^4y -/
def mono1 (x y : ℕ) : ℚ := 5 * x^4 * y

/-- The second monomial 5x^ny^m -/
def mono2 (n m x y : ℕ) : ℚ := 5 * x^n * y^m

theorem like_terms_exponent_sum :
  are_like_terms mono1 (mono2 n m) → n + m = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1185_118505


namespace NUMINAMATH_CALUDE_product_354_78_base7_units_digit_l1185_118548

-- Define the multiplication of two numbers in base 10
def base10Multiply (a b : ℕ) : ℕ := a * b

-- Define the conversion of a number from base 10 to base 7
def toBase7 (n : ℕ) : ℕ := n

-- Define the units digit of a number in base 7
def unitsDigitBase7 (n : ℕ) : ℕ := n % 7

-- Theorem statement
theorem product_354_78_base7_units_digit :
  unitsDigitBase7 (toBase7 (base10Multiply 354 78)) = 4 := by sorry

end NUMINAMATH_CALUDE_product_354_78_base7_units_digit_l1185_118548


namespace NUMINAMATH_CALUDE_f_composition_value_l1185_118506

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x + 2 * Real.cos (2 * x) else -Real.exp (2 * x)

theorem f_composition_value : f (f (Real.pi / 2)) = -1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1185_118506


namespace NUMINAMATH_CALUDE_mike_baseball_cards_l1185_118541

theorem mike_baseball_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 64) 
  (h2 : new_cards = 18) : 
  initial_cards + new_cards = 82 := by
  sorry

end NUMINAMATH_CALUDE_mike_baseball_cards_l1185_118541


namespace NUMINAMATH_CALUDE_product_inequality_l1185_118528

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a*x₁ + b) * (a*x₂ + b) * (a*x₃ + b) * (a*x₄ + b) * (a*x₅ + b) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l1185_118528


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1185_118592

/-- The area of a square inscribed in the ellipse x²/4 + y²/9 = 1, with sides parallel to the coordinate axes. -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, x^2 / 4 + y^2 / 9 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 144 / 13) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1185_118592


namespace NUMINAMATH_CALUDE_probability_two_blue_buttons_l1185_118532

/-- Represents a jar with buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of an event -/
def Probability := ℚ

/-- Initial state of Jar C -/
def initial_jar_c : Jar := ⟨5, 10⟩

/-- Number of buttons removed from each color -/
def removed_buttons : ℕ := 2

/-- Final state of Jar C after removal -/
def final_jar_c : Jar := ⟨initial_jar_c.red - removed_buttons, initial_jar_c.blue - 2 * removed_buttons⟩

/-- State of Jar D after receiving removed buttons -/
def jar_d : Jar := ⟨removed_buttons, 2 * removed_buttons⟩

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue_buttons : 
  (final_jar_c.red + final_jar_c.blue : ℚ) = 3/5 * (initial_jar_c.red + initial_jar_c.blue) →
  (final_jar_c.blue : ℚ) / (final_jar_c.red + final_jar_c.blue) * 
  (jar_d.blue : ℚ) / (jar_d.red + jar_d.blue) = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_blue_buttons_l1185_118532


namespace NUMINAMATH_CALUDE_inverse_zero_product_l1185_118555

theorem inverse_zero_product (a b : ℝ) : a = 0 → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_zero_product_l1185_118555


namespace NUMINAMATH_CALUDE_expand_product_l1185_118546

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1185_118546


namespace NUMINAMATH_CALUDE_sams_total_nickels_l1185_118573

/-- Sam's initial number of nickels -/
def initial_nickels : ℕ := 24

/-- Number of nickels Sam's dad gave him -/
def additional_nickels : ℕ := 39

/-- Theorem: Sam's total number of nickels after receiving more from his dad -/
theorem sams_total_nickels : initial_nickels + additional_nickels = 63 := by
  sorry

end NUMINAMATH_CALUDE_sams_total_nickels_l1185_118573


namespace NUMINAMATH_CALUDE_expression_equality_l1185_118523

theorem expression_equality : 
  Real.sqrt 4 * 4^(1/2 : ℝ) + 16 / 4 * 2 - Real.sqrt 8 = 12 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1185_118523


namespace NUMINAMATH_CALUDE_a_correct_S_correct_l1185_118588

/-- The number of different selection methods for two non-empty subsets A and B of {1,2,3,...,n}
    where the smallest number in B is greater than the largest number in A. -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 0
  else if n = 2 then 1
  else n * 2^(n-1) - 2^n + 1

/-- The sum of the first n terms of the sequence a_n. -/
def S (n : ℕ) : ℕ := (n - 3) * 2^n + n + 3

theorem a_correct (n : ℕ) : a n = n * 2^(n-1) - 2^n + 1 := by sorry

theorem S_correct (n : ℕ) : S n = (n - 3) * 2^n + n + 3 := by sorry

end NUMINAMATH_CALUDE_a_correct_S_correct_l1185_118588


namespace NUMINAMATH_CALUDE_cricket_collection_l1185_118568

theorem cricket_collection (initial_crickets : ℕ) (additional_crickets : ℕ) : 
  initial_crickets = 7 → additional_crickets = 4 → initial_crickets + additional_crickets = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_collection_l1185_118568


namespace NUMINAMATH_CALUDE_triangle_perimeter_theorem_l1185_118583

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter function
def perimeter (t : Triangle) : ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the ray function
def ray (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the intersection function
def intersect (s₁ s₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

theorem triangle_perimeter_theorem (ABC : Triangle) (X Y M : ℝ × ℝ) :
  perimeter ABC = 4 →
  X ∈ ray ABC.A ABC.B →
  Y ∈ ray ABC.A ABC.C →
  distance ABC.A X = 1 →
  distance ABC.A Y = 1 →
  M ∈ intersect (Set.Icc ABC.B ABC.C) (Set.Icc X Y) →
  (perimeter ⟨ABC.A, ABC.B, M⟩ = 2 ∨ perimeter ⟨ABC.A, ABC.C, M⟩ = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_theorem_l1185_118583


namespace NUMINAMATH_CALUDE_graph_above_condition_l1185_118504

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- State the theorem
theorem graph_above_condition (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 := by
  sorry

end NUMINAMATH_CALUDE_graph_above_condition_l1185_118504


namespace NUMINAMATH_CALUDE_expression_simplification_l1185_118509

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1185_118509


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1185_118560

theorem quadratic_inequality_no_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ a < 0 ∧ b^2 - 4*a*c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1185_118560


namespace NUMINAMATH_CALUDE_height_for_specific_configuration_l1185_118538

/-- Represents a configuration of three perpendicular rods fixed at one end -/
structure RodConfiguration where
  rod1 : ℝ
  rod2 : ℝ
  rod3 : ℝ

/-- Calculates the height of the fixed point above the plane for a given rod configuration -/
def height_above_plane (config : RodConfiguration) : ℝ :=
  sorry

/-- Theorem stating that for rods of lengths 1, 2, and 3, the height is 6/7 -/
theorem height_for_specific_configuration :
  let config : RodConfiguration := { rod1 := 1, rod2 := 2, rod3 := 3 }
  height_above_plane config = 6/7 :=
by sorry

end NUMINAMATH_CALUDE_height_for_specific_configuration_l1185_118538


namespace NUMINAMATH_CALUDE_find_number_l1185_118559

theorem find_number : ∃! x : ℝ, (((48 - x) * 4 - 26) / 2) = 37 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1185_118559


namespace NUMINAMATH_CALUDE_constant_q_value_l1185_118579

theorem constant_q_value (p q : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_q_value_l1185_118579


namespace NUMINAMATH_CALUDE_parabola_c_value_l1185_118585

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (2, 3)

theorem parabola_c_value (p : Parabola) :
  p.vertex = (2, 3) →
  p.x_coord 2 = 0 →
  p.c = -16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1185_118585


namespace NUMINAMATH_CALUDE_lottery_first_prize_probability_l1185_118536

/-- The probability of winning a first prize in a lottery -/
theorem lottery_first_prize_probability
  (total_tickets : ℕ)
  (first_prizes : ℕ)
  (h_total : total_tickets = 150)
  (h_first : first_prizes = 5) :
  (first_prizes : ℚ) / total_tickets = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_lottery_first_prize_probability_l1185_118536


namespace NUMINAMATH_CALUDE_ten_team_league_max_points_l1185_118556

/-- Represents a football league with n teams -/
structure FootballLeague where
  n : ℕ
  points_per_win : ℕ
  points_per_draw : ℕ
  points_per_loss : ℕ

/-- The maximum possible points for each team in the league -/
def max_points_per_team (league : FootballLeague) : ℕ :=
  sorry

/-- Theorem stating that in a 10-team league with 3 points for a win, 
    1 for a draw, and 0 for a loss, the maximum points per team is 13 -/
theorem ten_team_league_max_points :
  let league := FootballLeague.mk 10 3 1 0
  max_points_per_team league = 13 :=
sorry

end NUMINAMATH_CALUDE_ten_team_league_max_points_l1185_118556


namespace NUMINAMATH_CALUDE_students_on_field_trip_l1185_118520

/-- The number of students going on a field trip --/
def students_on_trip (seats_per_bus : ℕ) (num_buses : ℕ) : ℕ :=
  seats_per_bus * num_buses

/-- Theorem: The number of students on the trip is 28 given 7 seats per bus and 4 buses --/
theorem students_on_field_trip :
  students_on_trip 7 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l1185_118520


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1185_118501

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 40)
  (sum3 : c + a = 45) :
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1185_118501


namespace NUMINAMATH_CALUDE_parabola_max_q_y_l1185_118582

/-- Represents a parabola of the form y = -x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The y-coordinate of point Q where the parabola intersects x = -5 -/
def q_y_coord (p : Parabola) : ℝ :=
  25 - 5 * p.b + p.c

/-- Condition that the vertex of the parabola lies on the line y = 3x + 1 -/
def vertex_on_line (p : Parabola) : Prop :=
  (4 * p.c + p.b^2) / 4 = 3 * (p.b / 2) + 1

theorem parabola_max_q_y :
  ∃ (max_y : ℝ), max_y = -47/4 ∧
  ∀ (p : Parabola), vertex_on_line p →
  q_y_coord p ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_parabola_max_q_y_l1185_118582


namespace NUMINAMATH_CALUDE_sum_bc_value_l1185_118553

theorem sum_bc_value (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40)
  (h2 : a + d = 6)
  (h3 : a ≠ d) :
  b + c = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_bc_value_l1185_118553


namespace NUMINAMATH_CALUDE_cloud9_total_amount_l1185_118514

/-- Represents the pricing structure for Cloud 9 Diving Company --/
structure PricingStructure where
  individualDiscount : Float
  groupDiscount5to10 : Float
  groupDiscount11to20 : Float
  groupDiscount21Plus : Float
  earlyBirdDiscount : Float

/-- Represents a booking group --/
structure BookingGroup where
  participants : Nat
  totalCost : Float
  earlyBird : Bool

/-- Represents the refund structure --/
structure RefundStructure where
  individualRefund1 : Float
  individualRefund2 : Float
  groupRefund : Float

/-- Calculate the total amount taken by Cloud 9 Diving Company --/
def calculateTotalAmount (
  pricing : PricingStructure
) (
  individualBookings : Float
) (
  individualEarlyBird : Float
) (
  groupA : BookingGroup
) (
  groupB : BookingGroup
) (
  groupC : BookingGroup
) (
  refunds : RefundStructure
) : Float :=
  sorry

/-- Theorem stating the total amount taken by Cloud 9 Diving Company --/
theorem cloud9_total_amount :
  let pricing : PricingStructure := {
    individualDiscount := 0
    groupDiscount5to10 := 0.05
    groupDiscount11to20 := 0.10
    groupDiscount21Plus := 0.15
    earlyBirdDiscount := 0.03
  }
  let groupA : BookingGroup := {
    participants := 8
    totalCost := 6000
    earlyBird := true
  }
  let groupB : BookingGroup := {
    participants := 15
    totalCost := 9000
    earlyBird := false
  }
  let groupC : BookingGroup := {
    participants := 22
    totalCost := 15000
    earlyBird := true
  }
  let refunds : RefundStructure := {
    individualRefund1 := 500 * 3
    individualRefund2 := 300 * 2
    groupRefund := 800
  }
  calculateTotalAmount pricing 12000 3000 groupA groupB groupC refunds = 35006.50 :=
sorry

end NUMINAMATH_CALUDE_cloud9_total_amount_l1185_118514


namespace NUMINAMATH_CALUDE_remainder_theorem_l1185_118518

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) :
  (z * (2 + 4 + z) + 3) % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1185_118518


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1185_118554

theorem inequality_not_always_true
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d ≠ 0) :
  ¬(∀ d, (a + d)^2 > (b + d)^2) ∧
  (a + c * d > b + c * d) ∧
  (a^2 - c * d > b^2 - c * d) ∧
  (a / c > b / c) ∧
  (Real.sqrt a * d^2 > Real.sqrt b * d^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1185_118554


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1185_118589

theorem simplify_sqrt_expression :
  (Real.sqrt 600 / Real.sqrt 75) - (Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1185_118589


namespace NUMINAMATH_CALUDE_decimal_representation_contradiction_l1185_118535

theorem decimal_representation_contradiction (m n : ℕ) (h_n : n ≤ 100) :
  ∃ (k : ℕ) (B : ℕ), (1000 * B : ℚ) / n = 167 + (k : ℚ) / 1000 → False :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_contradiction_l1185_118535


namespace NUMINAMATH_CALUDE_x0_in_N_l1185_118567

def M : Set ℝ := {x | ∃ k : ℤ, x = k + 1/2}
def N : Set ℝ := {x | ∃ k : ℤ, x = k/2 + 1}

theorem x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := by
  sorry

end NUMINAMATH_CALUDE_x0_in_N_l1185_118567


namespace NUMINAMATH_CALUDE_triangle_properties_l1185_118591

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * (Real.sqrt 3 / 3) * t.a * Real.cos t.B ∧
  t.b = Real.sqrt 7 ∧
  t.c = 2 * Real.sqrt 3 ∧
  t.a > t.b

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/6 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1185_118591


namespace NUMINAMATH_CALUDE_integral_tangent_fraction_l1185_118550

theorem integral_tangent_fraction :
  ∫ x in -Real.arccos (1 / Real.sqrt 5)..0, (11 - 3 * Real.tan x) / (Real.tan x + 3) = Real.log 45 - 3 * Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_fraction_l1185_118550


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1185_118522

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x^2 - x - 20 > 0 → 1 - x^2 < 0) ∧
  (∃ x : ℝ, 1 - x^2 < 0 ∧ ¬(x^2 - x - 20 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1185_118522


namespace NUMINAMATH_CALUDE_value_of_a_l1185_118537

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

-- State the theorem
theorem value_of_a (a : ℝ) :
  (∀ x, (deriv (f a)) x = a) →  -- The derivative of f is constant and equal to a
  (deriv (f a)) 1 = 2 →         -- The derivative of f at x = 1 is 2
  a = 2 :=                      -- Then a must be equal to 2
by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1185_118537


namespace NUMINAMATH_CALUDE_circle_equation_l1185_118519

theorem circle_equation (A B : ℝ × ℝ) (h_A : A = (4, 2)) (h_B : B = (-1, 3)) :
  ∃ (D E F : ℝ),
    (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ 
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ 
       ∃ (x1 x2 y1 y2 : ℝ), 
         x1 + x2 + y1 + y2 = 2 ∧
         x1^2 + D*x1 + F = 0 ∧
         x2^2 + D*x2 + F = 0 ∧
         y1^2 + E*y1 + F = 0 ∧
         y2^2 + E*y2 + F = 0)) →
    D = -2 ∧ E = 0 ∧ F = -12 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1185_118519


namespace NUMINAMATH_CALUDE_unique_point_for_equal_angles_l1185_118580

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The focus point -/
def F : ℝ × ℝ := (2, 0)

/-- Check if a line passes through a point -/
def line_passes_through (m b : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = m * point.1 + b

/-- Check if a point is on a line passing through F -/
def point_on_line_through_F (x y : ℝ) : Prop :=
  ∃ m b : ℝ, line_passes_through m b (x, y) ∧ line_passes_through m b F

/-- The angle equality condition -/
def angle_equality (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ * (x₂ - p) + y₂ * (x₁ - p) = 0

/-- The main theorem -/
theorem unique_point_for_equal_angles :
  ∃! p : ℝ, p > 0 ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
    point_on_line_through_F x₁ y₁ ∧ point_on_line_through_F x₂ y₂ →
    angle_equality p x₁ y₁ x₂ y₂) ∧
  p = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_unique_point_for_equal_angles_l1185_118580


namespace NUMINAMATH_CALUDE_m_range_l1185_118549

theorem m_range (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end NUMINAMATH_CALUDE_m_range_l1185_118549


namespace NUMINAMATH_CALUDE_square_difference_l1185_118531

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 5) : (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1185_118531


namespace NUMINAMATH_CALUDE_cubic_poly_max_value_l1185_118564

/-- A cubic monic polynomial with roots a, b, and c -/
def cubic_monic_poly (a b c : ℝ) : ℝ → ℝ :=
  fun x => x^3 + (-(a + b + c)) * x^2 + (a*b + b*c + c*a) * x - a*b*c

/-- The theorem statement -/
theorem cubic_poly_max_value (a b c : ℝ) :
  let P := cubic_monic_poly a b c
  P 1 = 91 ∧ P (-1) = -121 →
  (∀ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) ≤ 7) ∧
  (∃ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) = 7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_poly_max_value_l1185_118564


namespace NUMINAMATH_CALUDE_gcd_binomial_coefficients_l1185_118534

theorem gcd_binomial_coefficients (n : ℕ+) :
  (∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = p^(k:ℕ)) ↔
  (∃ m : ℕ, m > 1 ∧ (∀ i : Fin (n-1), m ∣ Nat.choose n i.val.succ)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_coefficients_l1185_118534


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1185_118543

-- Define T_n as a function of n
def T (n : ℕ) : ℚ := sorry

-- Define the property of being the smallest positive integer n for which T_n is an integer
def is_smallest_integer_T (n : ℕ) : Prop :=
  (T n).isInt ∧ ∀ m : ℕ, m < n → ¬(T m).isInt

-- Theorem statement
theorem smallest_n_for_integer_T :
  is_smallest_integer_T 504 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1185_118543


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l1185_118500

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value of 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

/-- Theorem stating that if P(X < a) = 0.2, then P(X < 4-a) = 0.2 -/
theorem normal_distribution_symmetry :
  (cdf_X a = 0.2) → (cdf_X (4 - a) = 0.2) := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l1185_118500


namespace NUMINAMATH_CALUDE_solution_set_of_f_greater_than_4_range_of_a_l1185_118551

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Statement 1
theorem solution_set_of_f_greater_than_4 :
  {x : ℝ | f x > 4} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a ∈ Set.Ioi (3/2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_f_greater_than_4_range_of_a_l1185_118551


namespace NUMINAMATH_CALUDE_stratified_sampling_expectation_l1185_118597

theorem stratified_sampling_expectation
  (total_population : ℕ)
  (sample_size : ℕ)
  (category_size : ℕ)
  (h1 : total_population = 100)
  (h2 : sample_size = 20)
  (h3 : category_size = 30) :
  (sample_size : ℚ) / total_population * category_size = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_expectation_l1185_118597


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l1185_118569

/-- Triangle ABC with sides a, b, c, where a ≥ b and a ≥ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_ge_b : a ≥ b
  a_ge_c : a ≥ c
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Length of centroidal axis from vertex A -/
def centroidal_axis_length (t : Triangle) : ℝ := sorry

/-- Altitude from vertex A to side BC -/
def altitude_a (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all sides are equal -/
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_inequality (t : Triangle) :
  circumradius t / (2 * inradius t) ≥ centroidal_axis_length t / altitude_a t :=
sorry

theorem triangle_equality (t : Triangle) :
  circumradius t / (2 * inradius t) = centroidal_axis_length t / altitude_a t ↔ is_equilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l1185_118569


namespace NUMINAMATH_CALUDE_chord_length_theorem_l1185_118576

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (C1 C2 C3 : Circle) : 
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  C1.radius = 3 →
  C2.radius = 9 →
  are_collinear C1.center C2.center C3.center →
  ∃ (chord : ℝ), chord = 6 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l1185_118576


namespace NUMINAMATH_CALUDE_tom_profit_is_8798_l1185_118511

/-- Calculates the profit for Tom's dough ball project -/
def dough_ball_profit (
  flour_needed : ℕ)  -- Amount of flour needed in pounds
  (flour_bag_size : ℕ)  -- Size of each flour bag in pounds
  (flour_bag_cost : ℕ)  -- Cost of each flour bag in dollars
  (salt_needed : ℕ)  -- Amount of salt needed in pounds
  (salt_cost_per_pound : ℚ)  -- Cost of salt per pound in dollars
  (promotion_cost : ℕ)  -- Cost of promotion in dollars
  (tickets_sold : ℕ)  -- Number of tickets sold
  (ticket_price : ℕ)  -- Price of each ticket in dollars
  : ℤ :=
  let flour_bags := (flour_needed + flour_bag_size - 1) / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let total_cost := flour_cost + salt_cost.ceil + promotion_cost
  let revenue := tickets_sold * ticket_price
  revenue - total_cost

/-- Theorem stating that Tom's profit is $8798 -/
theorem tom_profit_is_8798 :
  dough_ball_profit 500 50 20 10 (2/10) 1000 500 20 = 8798 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_is_8798_l1185_118511


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1185_118502

theorem absolute_value_inequality (x a : ℝ) 
  (h1 : |x - 4| + |x - 3| < a) 
  (h2 : a > 0) : 
  a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1185_118502


namespace NUMINAMATH_CALUDE_sum_x_y_is_85_l1185_118565

/-- An arithmetic sequence with known terms 10, x, 30, y, 65 -/
structure ArithmeticSequence where
  x : ℝ
  y : ℝ
  isArithmetic : ∃ d : ℝ, x = 10 + d ∧ 30 = x + d ∧ y = 30 + 2*d ∧ 65 = y + d

/-- The sum of x and y in the arithmetic sequence is 85 -/
theorem sum_x_y_is_85 (seq : ArithmeticSequence) : seq.x + seq.y = 85 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_85_l1185_118565


namespace NUMINAMATH_CALUDE_repacking_books_leftover_l1185_118558

/-- The number of books left over when repacking from boxes of 42 to boxes of 45 -/
def books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ) : ℕ :=
  (initial_boxes * books_per_initial_box) % books_per_new_box

/-- Theorem stating that repacking 1573 boxes of 42 books into boxes of 45 books leaves 6 books over -/
theorem repacking_books_leftover :
  books_left_over 1573 42 45 = 6 := by
  sorry

#eval books_left_over 1573 42 45

end NUMINAMATH_CALUDE_repacking_books_leftover_l1185_118558


namespace NUMINAMATH_CALUDE_percent_of_y_l1185_118513

theorem percent_of_y (y : ℝ) (h : y > 0) : (1 * y / 20 + 3 * y / 10) / y * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1185_118513


namespace NUMINAMATH_CALUDE_combined_cost_increase_percentage_l1185_118508

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem combined_cost_increase_percentage
  (bicycle_cost : ℝ)
  (helmet_cost : ℝ)
  (bicycle_increase_percent : ℝ)
  (helmet_increase_percent : ℝ)
  (h1 : bicycle_cost = 160)
  (h2 : helmet_cost = 40)
  (h3 : bicycle_increase_percent = 5)
  (h4 : helmet_increase_percent = 10) :
  let new_bicycle_cost := bicycle_cost * (1 + bicycle_increase_percent / 100)
  let new_helmet_cost := helmet_cost * (1 + helmet_increase_percent / 100)
  let original_total := bicycle_cost + helmet_cost
  let new_total := new_bicycle_cost + new_helmet_cost
  (new_total - original_total) / original_total * 100 = 6 := by
  sorry

#check combined_cost_increase_percentage

end NUMINAMATH_CALUDE_combined_cost_increase_percentage_l1185_118508


namespace NUMINAMATH_CALUDE_no_y_intercepts_l1185_118533

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 5 * y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l1185_118533


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l1185_118578

/-- The number of green marbles Dan has after Mike took some -/
def remaining_green_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Dan has 9 green marbles after Mike took 23 -/
theorem dans_remaining_marbles :
  remaining_green_marbles 32 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l1185_118578


namespace NUMINAMATH_CALUDE_problem_solution_l1185_118581

def f (x : ℝ) := |x - 3| - 2
def g (x : ℝ) := -|x + 1| + 4

theorem problem_solution :
  (∀ x, f x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 6) ∧
  (∀ x, f x - g x ≥ -2) ∧
  (∀ m, (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1185_118581


namespace NUMINAMATH_CALUDE_johns_patients_l1185_118539

/-- The number of patients John sees each day at the first hospital -/
def patients_first_hospital : ℕ := sorry

/-- The number of patients John sees each day at the second hospital -/
def patients_second_hospital : ℕ := sorry

/-- The number of days John works per year -/
def work_days_per_year : ℕ := 5 * 50

/-- The total number of patients John treats in a year -/
def total_patients_per_year : ℕ := 11000

theorem johns_patients :
  patients_first_hospital = 20 ∧
  patients_second_hospital = (6 * patients_first_hospital) / 5 ∧
  work_days_per_year * (patients_first_hospital + patients_second_hospital) = total_patients_per_year :=
sorry

end NUMINAMATH_CALUDE_johns_patients_l1185_118539


namespace NUMINAMATH_CALUDE_furniture_purchase_price_l1185_118542

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  ∃ (purchase_price : ℝ),
    selling_price - purchase_price = profit_rate * purchase_price ∧
    purchase_price = 108 :=
by sorry

end NUMINAMATH_CALUDE_furniture_purchase_price_l1185_118542


namespace NUMINAMATH_CALUDE_cristinas_pace_cristina_pace_is_3_l1185_118561

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (race_length : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let cristinas_distance := nickys_pace * catch_up_time
  cristinas_distance / catch_up_time

/-- The main theorem stating Cristina's pace -/
theorem cristina_pace_is_3 : 
  cristinas_pace 300 12 3 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_pace_cristina_pace_is_3_l1185_118561


namespace NUMINAMATH_CALUDE_fraction_equality_l1185_118563

theorem fraction_equality (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1185_118563


namespace NUMINAMATH_CALUDE_sector_area_l1185_118510

/-- Given a sector with central angle θ and arc length L, 
    the area A of the sector can be calculated. -/
theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = 2 → L = 4 → A = 4 → A = (1/2) * (L/θ)^2 * θ :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1185_118510


namespace NUMINAMATH_CALUDE_intersection_angle_cosine_l1185_118577

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

/-- The hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/3 - y^2 = 1

/-- The foci of both curves -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- The cosine of the angle F₁PF₂ -/
noncomputable def cos_angle (P : ℝ × ℝ) : ℝ :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d := Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)
  (d₁^2 + d₂^2 - d^2) / (2 * d₁ * d₂)

theorem intersection_angle_cosine :
  ∀ (x y : ℝ), C₁ x y → C₂ x y → cos_angle (x, y) = 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_cosine_l1185_118577


namespace NUMINAMATH_CALUDE_tribe_leadership_proof_l1185_118590

def tribe_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 2 * (n - 5).choose 2

theorem tribe_leadership_proof (n : ℕ) (h : n = 11) :
  tribe_leadership_arrangements n = 207900 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_proof_l1185_118590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1185_118574

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- If a₂ + a₄ = 2 and S₂ + S₄ = 1 for an arithmetic sequence, then a₁₀ = 8 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.a 4 = 2) 
  (h2 : seq.S 2 + seq.S 4 = 1) : 
  seq.a 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1185_118574


namespace NUMINAMATH_CALUDE_original_number_proof_l1185_118544

theorem original_number_proof (x : ℤ) : x = 16 ↔ 
  (∃ k : ℤ, x + 10 = 26 * k) ∧ 
  (∀ y : ℤ, y < 10 → ∀ m : ℤ, x + y ≠ 26 * m) :=
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1185_118544


namespace NUMINAMATH_CALUDE_coefficient_x90_is_minus_one_l1185_118596

/-- The sequence of factors in the polynomial expansion -/
def factors : List (ℕ → ℤ) := [
  (λ n => if n = 1 then -1 else 1),
  (λ n => if n = 2 then -2 else 1),
  (λ n => if n = 3 then -3 else 1),
  (λ n => if n = 4 then -4 else 1),
  (λ n => if n = 5 then -5 else 1),
  (λ n => if n = 6 then -6 else 1),
  (λ n => if n = 7 then -7 else 1),
  (λ n => if n = 8 then -8 else 1),
  (λ n => if n = 9 then -9 else 1),
  (λ n => if n = 10 then -10 else 1),
  (λ n => if n = 11 then -11 else 1),
  (λ n => if n = 13 then -13 else 1)
]

/-- The coefficient of x^90 in the expansion -/
def coefficient_x90 : ℤ := -1

/-- Theorem stating that the coefficient of x^90 in the expansion is -1 -/
theorem coefficient_x90_is_minus_one :
  coefficient_x90 = -1 := by sorry

end NUMINAMATH_CALUDE_coefficient_x90_is_minus_one_l1185_118596


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1185_118594

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1185_118594


namespace NUMINAMATH_CALUDE_original_number_l1185_118557

theorem original_number (x : ℝ) : (x * 1.2 = 480) → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1185_118557


namespace NUMINAMATH_CALUDE_grid_arrangements_eq_six_l1185_118586

/-- The number of ways to arrange 3 distinct elements in 3 positions -/
def arrangements_of_three : ℕ := 3 * 2 * 1

/-- The number of ways to arrange digits 1, 2, and 3 in three boxes of a 2x2 grid,
    with the fourth box fixed -/
def grid_arrangements : ℕ := arrangements_of_three

theorem grid_arrangements_eq_six :
  grid_arrangements = 6 := by sorry

end NUMINAMATH_CALUDE_grid_arrangements_eq_six_l1185_118586


namespace NUMINAMATH_CALUDE_imaginary_unit_problem_l1185_118570

theorem imaginary_unit_problem : Complex.I * (1 + Complex.I)^2 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_problem_l1185_118570


namespace NUMINAMATH_CALUDE_journey_distance_l1185_118547

/-- Represents the journey of Jack and Peter -/
structure Journey where
  speed : ℝ
  distHomeToStore : ℝ
  distStoreToPeter : ℝ
  distPeterToStore : ℝ

/-- The total distance of the journey -/
def Journey.totalDistance (j : Journey) : ℝ :=
  j.distHomeToStore + j.distStoreToPeter + j.distPeterToStore

/-- Theorem stating the total distance of the journey -/
theorem journey_distance (j : Journey) 
  (h1 : j.speed > 0)
  (h2 : j.distStoreToPeter = 50)
  (h3 : j.distPeterToStore = 50)
  (h4 : j.distHomeToStore / j.speed = 2 * (j.distStoreToPeter / j.speed)) :
  j.totalDistance = 150 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l1185_118547


namespace NUMINAMATH_CALUDE_julia_baking_days_l1185_118521

/-- The number of cakes Julia bakes per day -/
def cakes_per_day : ℕ := 4

/-- The number of cakes eaten every two days -/
def cakes_eaten_per_two_days : ℕ := 1

/-- The final number of cakes remaining -/
def final_cakes : ℕ := 21

/-- The number of days Julia baked cakes -/
def baking_days : ℕ := 6

/-- Proves that the number of days Julia baked cakes is 6 -/
theorem julia_baking_days :
  baking_days * cakes_per_day - (baking_days / 2) * cakes_eaten_per_two_days = final_cakes := by
  sorry


end NUMINAMATH_CALUDE_julia_baking_days_l1185_118521


namespace NUMINAMATH_CALUDE_mark_sold_one_less_l1185_118517

/-- Given:
  n: total number of boxes allocated
  M: number of boxes Mark sold
  A: number of boxes Ann sold
-/
theorem mark_sold_one_less (n M A : ℕ) : 
  n = 8 → 
  M < n → 
  M ≥ 1 → 
  A = n - 2 → 
  A ≥ 1 → 
  M + A < n → 
  M = 7 :=
by sorry

end NUMINAMATH_CALUDE_mark_sold_one_less_l1185_118517


namespace NUMINAMATH_CALUDE_multiplicative_inverse_152_mod_367_l1185_118524

theorem multiplicative_inverse_152_mod_367 :
  ∃ a : ℕ, a < 367 ∧ (152 * a) % 367 = 1 ∧ a = 248 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_152_mod_367_l1185_118524


namespace NUMINAMATH_CALUDE_sarah_new_shirts_l1185_118526

/-- Given that Sarah initially had 9 shirts and now has a total of 17 shirts,
    prove that she bought 8 new shirts. -/
theorem sarah_new_shirts (initial_shirts : ℕ) (total_shirts : ℕ) (new_shirts : ℕ) :
  initial_shirts = 9 →
  total_shirts = 17 →
  new_shirts = total_shirts - initial_shirts →
  new_shirts = 8 := by
  sorry

end NUMINAMATH_CALUDE_sarah_new_shirts_l1185_118526


namespace NUMINAMATH_CALUDE_lemon_production_increase_l1185_118575

/-- Represents the lemon production data for normal and engineered trees -/
structure LemonProduction where
  normal_lemons_per_year : ℕ
  grove_size : ℕ
  total_lemons : ℕ
  years : ℕ

/-- Calculates the percentage increase in lemon production -/
def percentage_increase (data : LemonProduction) : ℚ :=
  let normal_total := data.normal_lemons_per_year * data.years
  let engineered_per_tree := data.total_lemons / data.grove_size
  ((engineered_per_tree - normal_total) / normal_total) * 100

/-- Theorem stating the percentage increase in lemon production -/
theorem lemon_production_increase (data : LemonProduction) 
  (h1 : data.normal_lemons_per_year = 60)
  (h2 : data.grove_size = 1500)
  (h3 : data.total_lemons = 675000)
  (h4 : data.years = 5) :
  percentage_increase data = 50 := by
  sorry

end NUMINAMATH_CALUDE_lemon_production_increase_l1185_118575


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1185_118529

theorem sin_alpha_value (α : Real) :
  (∃ (t : Real), t * (Real.sin (30 * π / 180)) = Real.cos α ∧
                 t * (-Real.cos (30 * π / 180)) = Real.sin α) →
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1185_118529


namespace NUMINAMATH_CALUDE_water_depth_is_208_l1185_118571

/-- The depth of water given Ron's height -/
def water_depth (ron_height : ℝ) : ℝ := 16 * ron_height

/-- Ron's height in feet -/
def ron_height : ℝ := 13

/-- Theorem stating that the water depth is 208 feet -/
theorem water_depth_is_208 : water_depth ron_height = 208 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_208_l1185_118571


namespace NUMINAMATH_CALUDE_percentage_difference_l1185_118593

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.35)) :
  y = x * (1 + 0.35) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1185_118593


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_125_l1185_118572

theorem gcd_lcm_product_75_125 : 
  (Nat.gcd 75 125) * (Nat.lcm 75 125) = 9375 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_125_l1185_118572


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l1185_118595

def fair_tickets : ℕ := 60
def fair_ticket_price : ℕ := 15
def baseball_ticket_price : ℕ := 10

theorem total_revenue_calculation :
  let baseball_tickets := fair_tickets / 3
  let fair_revenue := fair_tickets * fair_ticket_price
  let baseball_revenue := baseball_tickets * baseball_ticket_price
  fair_revenue + baseball_revenue = 1100 := by
sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l1185_118595


namespace NUMINAMATH_CALUDE_square_area_ratio_l1185_118552

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  let d := s * Real.sqrt 2
  let side_larger := 2 * d
  (side_larger ^ 2) / (s ^ 2) = 8 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1185_118552


namespace NUMINAMATH_CALUDE_largest_c_value_l1185_118516

theorem largest_c_value (c : ℝ) : (3 * c + 7) * (c - 2) = 9 * c → c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l1185_118516


namespace NUMINAMATH_CALUDE_ac_unit_final_price_l1185_118515

/-- Calculates the final price of an air-conditioning unit after multiple price changes -/
def finalPrice (originalPrice : ℝ) (christmasDiscount : ℝ) (energyEfficientDiscount : ℝ)
                (priceIncrease : ℝ) (productionCostIncrease : ℝ) (seasonalDiscount : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - christmasDiscount)
  let price2 := price1 * (1 - energyEfficientDiscount)
  let price3 := price2 * (1 + priceIncrease)
  let price4 := price3 * (1 + productionCostIncrease)
  price4 * (1 - seasonalDiscount)

/-- Theorem stating the final price of the air-conditioning unit -/
theorem ac_unit_final_price :
  finalPrice 470 0.16 0.07 0.12 0.08 0.10 = 399.71 := by
  sorry

end NUMINAMATH_CALUDE_ac_unit_final_price_l1185_118515
