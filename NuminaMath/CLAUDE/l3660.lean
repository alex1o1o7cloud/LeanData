import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3660_366055

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {-1, a^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (a : ℝ) : Prop := A a ∩ B = {4}

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -2 → property a) ∧
  ¬(∀ a : ℝ, property a → a = -2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3660_366055


namespace NUMINAMATH_CALUDE_probability_king_ace_standard_deck_l3660_366082

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (kings : Nat)
  (aces : Nat)

/-- The probability of drawing a King as the top card and an Ace as the second card -/
def probability_king_ace (d : Deck) : Rat :=
  (d.kings : Rat) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King as the top card and an Ace as the second card
    in a standard 52-card deck is 4/663 -/
theorem probability_king_ace_standard_deck :
  probability_king_ace ⟨52, 4, 4⟩ = 4 / 663 := by
  sorry

#eval probability_king_ace ⟨52, 4, 4⟩

end NUMINAMATH_CALUDE_probability_king_ace_standard_deck_l3660_366082


namespace NUMINAMATH_CALUDE_employee_share_l3660_366081

theorem employee_share (total_profit : ℝ) (num_employees : ℕ) (employer_percentage : ℝ) :
  total_profit = 50 ∧ num_employees = 9 ∧ employer_percentage = 0.1 →
  (total_profit - (employer_percentage * total_profit)) / num_employees = 5 := by
  sorry

end NUMINAMATH_CALUDE_employee_share_l3660_366081


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3660_366084

theorem sum_of_three_consecutive_cubes_divisible_by_nine (k : ℕ) :
  ∃ m : ℤ, k^3 + (k+1)^3 + (k+2)^3 = 9*m := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3660_366084


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3660_366016

theorem fifteenth_student_age 
  (total_students : Nat)
  (class_average : ℚ)
  (group1_size : Nat)
  (group1_average : ℚ)
  (group2_size : Nat)
  (group3_size : Nat)
  (group3_average : ℚ)
  (remaining_boys_average : ℚ)
  (h1 : total_students = 15)
  (h2 : class_average = 15.2)
  (h3 : group1_size = 5)
  (h4 : group1_average = 14)
  (h5 : group2_size = 4)
  (h6 : group3_size = 3)
  (h7 : group3_average = 16.6)
  (h8 : remaining_boys_average = 15.4)
  (h9 : total_students = group1_size + group2_size + group3_size + (total_students - group1_size - group2_size - group3_size))
  : ∃ (fifteenth_student_age : ℚ), fifteenth_student_age = 15.7 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3660_366016


namespace NUMINAMATH_CALUDE_paolo_coconuts_l3660_366037

theorem paolo_coconuts (paolo dante : ℕ) : 
  dante = 3 * paolo →  -- Dante has thrice as many coconuts as Paolo
  dante - 10 = 32 →    -- Dante had 32 coconuts left after selling 10
  paolo = 14 :=        -- Paolo had 14 coconuts
by
  sorry

end NUMINAMATH_CALUDE_paolo_coconuts_l3660_366037


namespace NUMINAMATH_CALUDE_age_puzzle_l3660_366064

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 18) (h2 : N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3660_366064


namespace NUMINAMATH_CALUDE_find_a_l3660_366096

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 5}

theorem find_a : ∃ a : ℝ, A ∩ B a = {x | 3 < x ∧ x < 7} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3660_366096


namespace NUMINAMATH_CALUDE_magnitude_of_B_area_of_triangle_l3660_366091

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangleCondition (t : Triangle) : Prop :=
  2 * t.b * Real.sin t.B = (2 * t.a + t.c) * Real.sin t.A + (2 * t.c + t.a) * Real.sin t.C

-- Theorem for part I
theorem magnitude_of_B (t : Triangle) (h : triangleCondition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem for part II
theorem area_of_triangle (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = Real.sqrt 3) (h3 : t.A = Real.pi / 4) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 - Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_B_area_of_triangle_l3660_366091


namespace NUMINAMATH_CALUDE_triangulation_labeling_exists_l3660_366057

/-- A convex polygon with n+1 vertices -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin (n+1) → ℝ × ℝ

/-- A triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  triangles : Fin (n-1) → Fin 3 → Fin (n+1)

/-- A labeling of triangles in a triangulation -/
def Labeling (n : ℕ) := Fin (n-1) → Fin (n-1)

/-- Predicate to check if a vertex is part of a triangle -/
def isVertexOfTriangle (n : ℕ) (t : Triangulation n) (v : Fin (n+1)) (tri : Fin (n-1)) : Prop :=
  ∃ i : Fin 3, t.triangles tri i = v

/-- Main theorem statement -/
theorem triangulation_labeling_exists (n : ℕ) (t : Triangulation n) :
  ∃ l : Labeling n, ∀ i : Fin (n-1), isVertexOfTriangle n t i (l i) :=
sorry

end NUMINAMATH_CALUDE_triangulation_labeling_exists_l3660_366057


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3660_366006

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the expansion of (x-1)^6
def expansion_x_minus_1_power_6 (r : ℕ) : ℤ := binomial 6 r * (-1)^r

-- Theorem statement
theorem coefficient_x_squared_in_expansion :
  expansion_x_minus_1_power_6 3 = -20 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3660_366006


namespace NUMINAMATH_CALUDE_value_of_fraction_difference_l3660_366029

theorem value_of_fraction_difference (x y : ℝ) 
  (hx : x = Real.sqrt 5 - 1) 
  (hy : y = Real.sqrt 5 + 1) : 
  1 / x - 1 / y = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_value_of_fraction_difference_l3660_366029


namespace NUMINAMATH_CALUDE_range_of_a_l3660_366097

def f (x a : ℝ) : ℝ := |x - a| + x + 5

theorem range_of_a (a : ℝ) : (∀ x, f x a ≥ 8) ↔ |a + 5| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3660_366097


namespace NUMINAMATH_CALUDE_circus_investment_revenue_l3660_366073

/-- A circus production investment problem -/
theorem circus_investment_revenue (overhead : ℕ) (production_cost : ℕ) (break_even_performances : ℕ) :
  overhead = 81000 →
  production_cost = 7000 →
  break_even_performances = 9 →
  (overhead + break_even_performances * production_cost) / break_even_performances = 16000 :=
by sorry

end NUMINAMATH_CALUDE_circus_investment_revenue_l3660_366073


namespace NUMINAMATH_CALUDE_characterization_of_complete_sets_l3660_366050

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

def complete_sets : Set (Set ℕ) :=
  {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, Set.univ}

theorem characterization_of_complete_sets :
  ∀ A : Set ℕ, A.Nonempty → (is_complete A ↔ A ∈ complete_sets) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_complete_sets_l3660_366050


namespace NUMINAMATH_CALUDE_marias_reading_capacity_l3660_366056

/-- Given Maria's reading speed and available time, prove how many complete books she can read --/
theorem marias_reading_capacity (pages_per_hour : ℕ) (book_pages : ℕ) (available_hours : ℕ) : 
  pages_per_hour = 120 → book_pages = 360 → available_hours = 8 → 
  (available_hours * pages_per_hour) / book_pages = 2 := by
  sorry

#check marias_reading_capacity

end NUMINAMATH_CALUDE_marias_reading_capacity_l3660_366056


namespace NUMINAMATH_CALUDE_chord_equation_l3660_366087

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of a line containing a chord of an ellipse is 2x + y - 4 = 0 -/
theorem chord_equation (m n s t : ℝ) (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
  (h_sum : m + n = 3)
  (h_frac : m / s + n / t = 1)
  (h_order : m < n)
  (h_min : ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s' + t' ≥ 3 + 2 * Real.sqrt 2)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 / 4 + y₁^2 / 16 = 1 ∧
    x₂^2 / 4 + y₂^2 / 16 = 1 ∧
    (x₁ + x₂) / 2 = m ∧
    (y₁ + y₂) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ 2 * a + b = 0 ∧ a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_chord_equation_l3660_366087


namespace NUMINAMATH_CALUDE_linear_independence_preservation_l3660_366053

variable {n : ℕ}
variable (v : Fin (n - 1) → (Fin n → ℝ))

/-- P_{i,k} sets the i-th component of a vector to zero -/
def P (i k : ℕ) (x : Fin k → ℝ) : Fin k → ℝ :=
  λ j => if j = i then 0 else x j

theorem linear_independence_preservation (hn : n ≥ 2) 
  (hv : LinearIndependent ℝ v) :
  ∃ k : Fin n, LinearIndependent ℝ (λ i => P k n (v i)) := by
  sorry

end NUMINAMATH_CALUDE_linear_independence_preservation_l3660_366053


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3660_366075

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 3) / 2
  let x₂ : ℝ := (3 - Real.sqrt 3) / 2
  2 * x₁^2 - 6 * x₁ + 3 = 0 ∧ 2 * x₂^2 - 6 * x₂ + 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3660_366075


namespace NUMINAMATH_CALUDE_C_sufficient_for_A_l3660_366034

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the conditions
variable (h1 : A ↔ B)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem statement
theorem C_sufficient_for_A : C → A := by
  sorry

end NUMINAMATH_CALUDE_C_sufficient_for_A_l3660_366034


namespace NUMINAMATH_CALUDE_workplace_distance_l3660_366083

/-- Calculates the one-way distance to a workplace given the speed and total round trip time -/
theorem workplace_distance (speed : ℝ) (total_time : ℝ) : 
  speed = 40 → total_time = 3 → (speed * total_time) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_workplace_distance_l3660_366083


namespace NUMINAMATH_CALUDE_youngest_child_age_possibility_l3660_366039

/-- Represents the ages of the four children -/
structure ChildrenAges where
  twin : ℕ
  other1 : ℕ
  other2 : ℕ

/-- The problem statement -/
theorem youngest_child_age_possibility (fatherAge : ℕ) (fatherCharge : ℚ) (childCharge : ℚ) (totalBill : ℚ) :
  fatherAge = 30 →
  fatherCharge = 1/4 →
  childCharge = 11/20 →
  totalBill = 151/10 →
  ∃ (ages : ChildrenAges), 
    (ages.twin * 2 + ages.other1 + ages.other2 : ℚ) * childCharge + fatherAge * fatherCharge = totalBill ∧
    ages.other1 ≠ ages.other2 ∧
    min ages.twin (min ages.other1 ages.other2) = 1 :=
sorry


end NUMINAMATH_CALUDE_youngest_child_age_possibility_l3660_366039


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3660_366019

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3660_366019


namespace NUMINAMATH_CALUDE_rug_purchase_price_l3660_366051

/-- Proves that the purchase price per rug is $40 given the selling price, number of rugs, and total profit -/
theorem rug_purchase_price
  (selling_price : ℝ)
  (num_rugs : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 60)
  (h2 : num_rugs = 20)
  (h3 : total_profit = 400) :
  (selling_price * num_rugs - total_profit) / num_rugs = 40 := by
  sorry

end NUMINAMATH_CALUDE_rug_purchase_price_l3660_366051


namespace NUMINAMATH_CALUDE_jean_card_money_l3660_366071

/-- The amount of money Jean puts in each card for her grandchildren --/
def money_per_card (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (total_money : ℕ) : ℚ :=
  total_money / (num_grandchildren * cards_per_grandchild)

/-- Theorem: Jean puts $80 in each card for her grandchildren --/
theorem jean_card_money :
  money_per_card 3 2 480 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jean_card_money_l3660_366071


namespace NUMINAMATH_CALUDE_inequality_proof_l3660_366025

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3660_366025


namespace NUMINAMATH_CALUDE_football_players_count_l3660_366062

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : softball_players = 19)
  (h4 : total_players = 77) :
  total_players - (cricket_players + hockey_players + softball_players) = 21 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l3660_366062


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3660_366015

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - cubeRoot 8) * (x - cubeRoot 27) * (x - cubeRoot 64) = 1

-- Define the roots
noncomputable def u : ℝ := sorry
noncomputable def v : ℝ := sorry
noncomputable def w : ℝ := sorry

-- State the theorem
theorem sum_of_cubes_of_roots :
  equation u ∧ equation v ∧ equation w →
  u^3 + v^3 + w^3 = 102 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3660_366015


namespace NUMINAMATH_CALUDE_cans_of_frosting_needed_l3660_366026

/-- The number of cans of frosting Bob needs to frost the remaining cakes -/
theorem cans_of_frosting_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (cans_per_cake : ℕ) : 
  cakes_per_day = 10 → days = 5 → cakes_eaten = 12 → cans_per_cake = 2 →
  (cakes_per_day * days - cakes_eaten) * cans_per_cake = 76 := by sorry

end NUMINAMATH_CALUDE_cans_of_frosting_needed_l3660_366026


namespace NUMINAMATH_CALUDE_break_room_capacity_l3660_366023

/-- The number of people that can be seated at each table -/
def people_per_table : ℕ := 8

/-- The number of tables in the break room -/
def number_of_tables : ℕ := 4

/-- The total number of people that can be seated in the break room -/
def total_seating_capacity : ℕ := people_per_table * number_of_tables

theorem break_room_capacity : total_seating_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_break_room_capacity_l3660_366023


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l3660_366076

theorem added_number_after_doubling (x : ℝ) : 
  3 * (2 * 7 + x) = 69 → x = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l3660_366076


namespace NUMINAMATH_CALUDE_inequality_proof_l3660_366042

theorem inequality_proof (A B C a b c r : ℝ) 
  (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  (A + a + B + b) / (A + a + B + b + c + r) + 
  (B + b + C + c) / (B + b + C + c + a + r) > 
  (c + c + A + a) / (C + c + A + a + b + r) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3660_366042


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3660_366014

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3660_366014


namespace NUMINAMATH_CALUDE_existence_of_integer_roots_l3660_366048

theorem existence_of_integer_roots : ∃ (a b c d e f : ℤ),
  (∀ x : ℤ, (x + a) * (x^2 + b*x + c) * (x^3 + d*x^2 + e*x + f) = 0 ↔ 
    x = a ∨ x^2 + b*x + c = 0 ∨ x^3 + d*x^2 + e*x + f = 0) ∧
  (∃! (r₁ r₂ r₃ r₄ r₅ r₆ : ℤ), 
    {r₁, r₂, r₃, r₄, r₅, r₆} = {a} ∪ 
      {x : ℤ | x^2 + b*x + c = 0} ∪ 
      {x : ℤ | x^3 + d*x^2 + e*x + f = 0}) :=
sorry

end NUMINAMATH_CALUDE_existence_of_integer_roots_l3660_366048


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l3660_366044

theorem continued_fraction_solution : 
  ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l3660_366044


namespace NUMINAMATH_CALUDE_shaded_area_is_65_l3660_366020

/-- Represents a trapezoid with a line segment dividing it into two parts -/
structure DividedTrapezoid where
  total_area : ℝ
  dividing_segment_length : ℝ
  inner_segment_length : ℝ

/-- Calculates the area of the shaded region in the divided trapezoid -/
def shaded_area (t : DividedTrapezoid) : ℝ :=
  t.total_area - (t.dividing_segment_length * t.inner_segment_length)

/-- Theorem stating that for the given trapezoid, the shaded area is 65 -/
theorem shaded_area_is_65 (t : DividedTrapezoid) 
  (h1 : t.total_area = 117)
  (h2 : t.dividing_segment_length = 13)
  (h3 : t.inner_segment_length = 4) :
  shaded_area t = 65 := by
  sorry

#eval shaded_area { total_area := 117, dividing_segment_length := 13, inner_segment_length := 4 }

end NUMINAMATH_CALUDE_shaded_area_is_65_l3660_366020


namespace NUMINAMATH_CALUDE_no_isosceles_triangles_l3660_366074

-- Define a point on a 2D grid
structure Point where
  x : Int
  y : Int

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Check if a triangle is isosceles
def isIsosceles (t : Triangle) : Bool :=
  let d1 := squaredDistance t.a t.b
  let d2 := squaredDistance t.b t.c
  let d3 := squaredDistance t.c t.a
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨2, 7⟩, ⟨5, 7⟩, ⟨5, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 2⟩, ⟨7, 2⟩, ⟨4, 6⟩⟩
def triangle3 : Triangle := ⟨⟨2, 1⟩, ⟨2, 4⟩, ⟨7, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨9, 8⟩, ⟨9, 9⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨8, 5⟩, ⟨10, 1⟩⟩

-- Theorem: None of the given triangles are isosceles
theorem no_isosceles_triangles : 
  ¬(isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ 
    isIsosceles triangle4 ∨ isIsosceles triangle5) := by
  sorry

end NUMINAMATH_CALUDE_no_isosceles_triangles_l3660_366074


namespace NUMINAMATH_CALUDE_greek_cross_dissection_l3660_366031

/-- Represents a Greek cross -/
structure GreekCross where
  area : ℝ
  squares : Fin 5 → Square

/-- Represents a square piece of a Greek cross -/
structure Square where
  side_length : ℝ

/-- Represents a piece obtained from cutting a Greek cross -/
inductive Piece
| Square : Square → Piece
| Composite : List Square → Piece

/-- Theorem stating that a Greek cross can be dissected into 12 pieces 
    to form three identical smaller Greek crosses -/
theorem greek_cross_dissection (original : GreekCross) :
  ∃ (pieces : List Piece) (small_crosses : Fin 3 → GreekCross),
    (pieces.length = 12) ∧
    (∀ i : Fin 3, (small_crosses i).area = original.area / 3) ∧
    (∀ i j : Fin 3, i ≠ j → small_crosses i = small_crosses j) ∧
    (∃ (reassembly : List Piece → Fin 3 → GreekCross), 
      reassembly pieces = small_crosses) :=
sorry

end NUMINAMATH_CALUDE_greek_cross_dissection_l3660_366031


namespace NUMINAMATH_CALUDE_triangle_radii_product_l3660_366007

theorem triangle_radii_product (a b c : ℝ) (ha : a = 26) (hb : b = 28) (hc : c = 30) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  R * r = 130 := by sorry

end NUMINAMATH_CALUDE_triangle_radii_product_l3660_366007


namespace NUMINAMATH_CALUDE_minimum_distance_theorem_l3660_366066

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

def line_l (x y : ℝ) : Prop := x + 2 * y - 2 * Real.log 2 - 6 = 0

def M (px py qx qy : ℝ) : ℝ := (px - qx)^2 + (py - qy)^2

theorem minimum_distance_theorem (px py qx qy : ℝ) 
  (h1 : f px = py) 
  (h2 : line_l qx qy) : 
  (∃ (min_M : ℝ), ∀ (px' py' qx' qy' : ℝ), 
    f px' = py' → line_l qx' qy' → 
    M px' py' qx' qy' ≥ min_M ∧ 
    min_M = 16/5 ∧
    (M px py qx qy = min_M → qx = 14/5)) := by sorry

end NUMINAMATH_CALUDE_minimum_distance_theorem_l3660_366066


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l3660_366072

theorem rectangle_area_diagonal_relation :
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  length / width = 5 / 2 →
  2 * (length + width) = 56 →
  ∃ (d : ℝ),
  d^2 = length^2 + width^2 ∧
  length * width = (10/29) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l3660_366072


namespace NUMINAMATH_CALUDE_infinitely_many_fantastic_triplets_l3660_366013

/-- Definition of a fantastic triplet -/
def is_fantastic_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℚ, b = k * a ∧ c = k * b) ∧
  (∃ d : ℤ, b + 1 - a = d ∧ c - (b + 1) = d)

/-- There exist infinitely many fantastic triplets -/
theorem infinitely_many_fantastic_triplets :
  ∀ i : ℕ, ∃ a b c : ℕ,
    is_fantastic_triplet a b c ∧
    a = 2^(2*i+1) ∧
    b = 2^(2*i+1) + 2^i ∧
    c = 2^(2*i+1) + 2^(i+2) + 2 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_fantastic_triplets_l3660_366013


namespace NUMINAMATH_CALUDE_integer_partition_impossibility_l3660_366001

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set ℤ), 
    (∀ (n : ℤ), (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
                (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
                (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
                (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
                (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
                (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)) ∧
    (A ∪ B ∪ C = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_integer_partition_impossibility_l3660_366001


namespace NUMINAMATH_CALUDE_farm_entrance_fee_for_students_l3660_366033

theorem farm_entrance_fee_for_students :
  let num_students : ℕ := 35
  let num_adults : ℕ := 4
  let adult_fee : ℚ := 6
  let total_cost : ℚ := 199
  let student_fee : ℚ := (total_cost - num_adults * adult_fee) / num_students
  student_fee = 5 := by sorry

end NUMINAMATH_CALUDE_farm_entrance_fee_for_students_l3660_366033


namespace NUMINAMATH_CALUDE_calvin_insect_collection_l3660_366041

/-- Calculates the total number of insects in Calvin's collection --/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  let beetles := 4 * crickets
  let other_insects := roaches + scorpions + crickets + caterpillars + beetles
  let exotic_insects := 3 * other_insects
  other_insects + exotic_insects

/-- Theorem stating that Calvin has 204 insects in his collection --/
theorem calvin_insect_collection : total_insects 12 3 = 204 := by
  sorry

end NUMINAMATH_CALUDE_calvin_insect_collection_l3660_366041


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3660_366000

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3660_366000


namespace NUMINAMATH_CALUDE_alternating_series_sum_equals_minus_30_l3660_366030

def alternatingSeriesSum (a₁ : ℤ) (d : ℤ) (lastTerm : ℤ) : ℤ :=
  -- Definition of the sum of the alternating series
  sorry

theorem alternating_series_sum_equals_minus_30 :
  alternatingSeriesSum 2 6 59 = -30 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_equals_minus_30_l3660_366030


namespace NUMINAMATH_CALUDE_probability_independent_of_shape_l3660_366086

/-- A geometric model related to area -/
structure GeometricModel where
  area : ℝ
  shape : Type

/-- The probability of a geometric model -/
def probability (model : GeometricModel) : ℝ := sorry

theorem probability_independent_of_shape (model1 model2 : GeometricModel) 
  (h : model1.area = model2.area) : 
  probability model1 = probability model2 := by sorry

end NUMINAMATH_CALUDE_probability_independent_of_shape_l3660_366086


namespace NUMINAMATH_CALUDE_smallest_m_perfect_square_and_cube_l3660_366036

theorem smallest_m_perfect_square_and_cube : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (k : ℕ), 5 * m = k * k) ∧ 
  (∃ (l : ℕ), 3 * m = l * l * l) ∧ 
  (∀ (n : ℕ), n > 0 → 
    (∃ (k : ℕ), 5 * n = k * k) → 
    (∃ (l : ℕ), 3 * n = l * l * l) → 
    m ≤ n) ∧
  m = 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_perfect_square_and_cube_l3660_366036


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3660_366088

theorem arithmetic_mean_of_fractions :
  (5/6 : ℚ) = (7/9 + 8/9) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3660_366088


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l3660_366095

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m - 1 > 0 ∧ m + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l3660_366095


namespace NUMINAMATH_CALUDE_probability_calculation_l3660_366061

/-- The probability of selecting exactly 2 purple and 2 orange marbles -/
def probability_two_purple_two_orange : ℚ :=
  66 / 1265

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 12

/-- The number of orange marbles in the bag -/
def orange_marbles : ℕ := 5

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles + orange_marbles

/-- The number of marbles selected -/
def selected_marbles : ℕ := 4

theorem probability_calculation :
  probability_two_purple_two_orange = 
    (Nat.choose purple_marbles 2 * Nat.choose orange_marbles 2) / 
    Nat.choose total_marbles selected_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l3660_366061


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l3660_366099

theorem largest_four_digit_divisible_by_12 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l3660_366099


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3660_366070

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3660_366070


namespace NUMINAMATH_CALUDE_min_dot_product_ellipse_l3660_366005

/-- The minimum dot product of OP and FP for an ellipse -/
theorem min_dot_product_ellipse :
  ∀ (x y : ℝ), 
  x^2 / 9 + y^2 / 8 = 1 →
  ∃ (min : ℝ), 
  (∀ (x' y' : ℝ), x'^2 / 9 + y'^2 / 8 = 1 → 
    x'^2 + x' + y'^2 ≥ min) ∧
  min = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_ellipse_l3660_366005


namespace NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l3660_366017

theorem square_side_length_equal_perimeter (r : ℝ) (s : ℝ) :
  r = 3 →  -- radius of the circle is 3 units
  4 * s = 2 * Real.pi * r →  -- perimeters are equal
  s = 3 * Real.pi / 2 :=  -- side length of the square
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l3660_366017


namespace NUMINAMATH_CALUDE_initial_men_count_l3660_366009

/-- Proves that the initial number of men is 1000, given the conditions of the problem. -/
theorem initial_men_count (initial_days : ℝ) (joined_days : ℝ) (joined_men : ℕ) : 
  initial_days = 20 →
  joined_days = 16.67 →
  joined_men = 200 →
  (∃ (initial_men : ℕ), initial_men * initial_days = (initial_men + joined_men) * joined_days ∧ initial_men = 1000) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l3660_366009


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l3660_366004

theorem no_quadratic_trinomial_with_odd_coeffs_and_2022th_root :
  ¬ ∃ (a b c : ℤ), 
    (Odd a ∧ Odd b ∧ Odd c) ∧ 
    (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l3660_366004


namespace NUMINAMATH_CALUDE_dot_product_bounds_l3660_366052

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Define the circle
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 + 1)^2 + Q.2^2 = 1

-- Define a tangent line from a point to the circle
def is_tangent (P A : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ ((P.1 - A.1) * (A.1 + 1) + (P.2 - A.2) * A.2 = 0)

-- Define the dot product of two vectors
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

-- The main theorem
theorem dot_product_bounds (P A B : ℝ × ℝ) :
  is_on_ellipse P → is_tangent P A → is_tangent P B →
  2 * Real.sqrt 2 - 3 ≤ dot_product P A B ∧ dot_product P A B ≤ 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_bounds_l3660_366052


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_folded_equilateral_triangle_l3660_366085

/-- The surface area of the circumscribed sphere of a tetrahedron formed by folding an equilateral triangle --/
theorem circumscribed_sphere_area_folded_equilateral_triangle :
  let side_length : ℝ := 2
  let height : ℝ := Real.sqrt 3
  let tetrahedron_edge1 : ℝ := 1
  let tetrahedron_edge2 : ℝ := 1
  let tetrahedron_edge3 : ℝ := height
  let sphere_radius : ℝ := Real.sqrt 5 / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 5 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_circumscribed_sphere_area_folded_equilateral_triangle_l3660_366085


namespace NUMINAMATH_CALUDE_sine_equality_solution_l3660_366054

theorem sine_equality_solution (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (780 * π / 180) → 
  m = 60 ∨ m = 120 := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_solution_l3660_366054


namespace NUMINAMATH_CALUDE_factor_expression_l3660_366038

theorem factor_expression (a b c : ℝ) :
  a^4*(b^3 - c^3) + b^4*(c^3 - a^3) + c^4*(a^3 - b^3) = 
  (a-b)*(b-c)*(c-a)*(a^2 + a*b + a*c + b^2 + b*c + c^2) := by
sorry

end NUMINAMATH_CALUDE_factor_expression_l3660_366038


namespace NUMINAMATH_CALUDE_largest_non_representable_number_l3660_366024

theorem largest_non_representable_number : ∃ (n : ℕ), n > 0 ∧
  (∀ x y : ℕ, x > 0 → y > 0 → 9 * x + 11 * y ≠ n) ∧
  (∀ m : ℕ, m > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = m) ∧
  (∀ k : ℕ, k > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = k) →
  n = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_non_representable_number_l3660_366024


namespace NUMINAMATH_CALUDE_ratio_w_y_is_15_4_l3660_366078

-- Define the ratios as fractions
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 5 / 3
def ratio_z_x : ℚ := 1 / 5

-- Theorem statement
theorem ratio_w_y_is_15_4 :
  let ratio_w_y := ratio_w_x / (ratio_y_z * ratio_z_x)
  ratio_w_y = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_y_is_15_4_l3660_366078


namespace NUMINAMATH_CALUDE_coin_difference_is_ten_l3660_366022

def coin_values : List ℕ := [5, 10, 25, 50]
def target_amount : ℕ := 60

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry
def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference_is_ten :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 10 := by sorry

end NUMINAMATH_CALUDE_coin_difference_is_ten_l3660_366022


namespace NUMINAMATH_CALUDE_parallelogram_opposite_sides_parallel_equal_l3660_366060

-- Define a parallelogram
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)
  (is_parallelogram : 
    (vertices 0 - vertices 1 = vertices 3 - vertices 2) ∧
    (vertices 0 - vertices 3 = vertices 1 - vertices 2))

-- Define the property of having parallel and equal opposite sides
def has_parallel_equal_opposite_sides (p : Parallelogram) : Prop :=
  (p.vertices 0 - p.vertices 1 = p.vertices 3 - p.vertices 2) ∧
  (p.vertices 0 - p.vertices 3 = p.vertices 1 - p.vertices 2)

-- Theorem stating that all parallelograms have parallel and equal opposite sides
theorem parallelogram_opposite_sides_parallel_equal (p : Parallelogram) :
  has_parallel_equal_opposite_sides p :=
by
  sorry

-- Note: Rectangles, rhombuses, and squares are special cases of parallelograms,
-- so this theorem applies to them as well.

end NUMINAMATH_CALUDE_parallelogram_opposite_sides_parallel_equal_l3660_366060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3660_366090

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3660_366090


namespace NUMINAMATH_CALUDE_y1_value_l3660_366098

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1 - y1)^2 + 2*(y1 - y2)^2 + 2*(y2 - y3)^2 + y3^2 = 1/2) :
  y1 = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_y1_value_l3660_366098


namespace NUMINAMATH_CALUDE_point_below_line_range_l3660_366092

/-- Given a point (-2,t) located below the line 2x-3y+6=0, prove that the range of t is (-∞, 2/3) -/
theorem point_below_line_range (t : ℝ) : 
  (2 * (-2) - 3 * t + 6 > 0) → (t < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_point_below_line_range_l3660_366092


namespace NUMINAMATH_CALUDE_smallest_k_value_l3660_366021

theorem smallest_k_value (x y : ℤ) (h1 : x = -2) (h2 : y = 5) : 
  ∃ k : ℤ, (∀ m : ℤ, k * x + 2 * y ≤ 4 → m * x + 2 * y ≤ 4 → k ≤ m) ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3660_366021


namespace NUMINAMATH_CALUDE_pizza_consumption_l3660_366093

theorem pizza_consumption (n : ℕ) (first_trip : ℚ) (subsequent_trips : ℚ) : 
  n = 6 → 
  first_trip = 2/3 → 
  subsequent_trips = 1/2 → 
  (1 - (1 - first_trip) * subsequent_trips^(n-1) : ℚ) = 191/192 := by
  sorry

end NUMINAMATH_CALUDE_pizza_consumption_l3660_366093


namespace NUMINAMATH_CALUDE_positive_difference_l3660_366035

theorem positive_difference (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  d - c - b - a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_l3660_366035


namespace NUMINAMATH_CALUDE_supporting_pillars_concrete_l3660_366008

/-- The amount of concrete needed for a bridge construction --/
structure BridgeConcrete where
  roadwayDeck : ℕ
  oneAnchor : ℕ
  totalBridge : ℕ

/-- Calculates the amount of concrete needed for supporting pillars --/
def supportingPillarsAmount (b : BridgeConcrete) : ℕ :=
  b.totalBridge - (b.roadwayDeck + 2 * b.oneAnchor)

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem supporting_pillars_concrete (b : BridgeConcrete) 
  (h1 : b.roadwayDeck = 1600)
  (h2 : b.oneAnchor = 700)
  (h3 : b.totalBridge = 4800) :
  supportingPillarsAmount b = 1800 := by
  sorry

#eval supportingPillarsAmount ⟨1600, 700, 4800⟩

end NUMINAMATH_CALUDE_supporting_pillars_concrete_l3660_366008


namespace NUMINAMATH_CALUDE_fraction_inequality_l3660_366011

theorem fraction_inequality (x : ℝ) (h : x ≠ 2) :
  (x + 1) / (x - 2) ≥ 0 ↔ x ≤ -1 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3660_366011


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_is_14_l3660_366067

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in a triangle 
    with sides 6, 8, and 10 is 14 -/
theorem sum_longest_altitudes_is_14 (t : RightTriangle) : 
  sum_longest_altitudes t = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_is_14_l3660_366067


namespace NUMINAMATH_CALUDE_rectangle_fourth_vertex_l3660_366043

-- Define a structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a structure for a rectangle
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the theorem
theorem rectangle_fourth_vertex 
  (ABCD : Rectangle)
  (h1 : ABCD.A = ⟨0, 1⟩)
  (h2 : ABCD.B = ⟨1, 0⟩)
  (h3 : ABCD.C = ⟨3, 2⟩)
  : ABCD.D = ⟨2, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fourth_vertex_l3660_366043


namespace NUMINAMATH_CALUDE_weighted_average_theorem_l3660_366012

def score1 : Rat := 55 / 100
def score2 : Rat := 67 / 100
def score3 : Rat := 76 / 100
def score4 : Rat := 82 / 100
def score5 : Rat := 85 / 100
def score6 : Rat := 48 / 60
def score7 : Rat := 150 / 200

def convertedScore6 : Rat := score6 * 100 / 60
def convertedScore7 : Rat := score7 * 100 / 200

def totalScores : Rat := score1 + score2 + score3 + score4 + score5 + convertedScore6 + convertedScore7
def numberOfScores : Nat := 7

theorem weighted_average_theorem :
  totalScores / numberOfScores = (55 + 67 + 76 + 82 + 85 + 80 + 75) / 7 := by sorry

end NUMINAMATH_CALUDE_weighted_average_theorem_l3660_366012


namespace NUMINAMATH_CALUDE_train_length_l3660_366069

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ length : ℝ, 
  (length ≥ 150 ∧ length < 151) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3660_366069


namespace NUMINAMATH_CALUDE_parallel_vector_with_given_magnitude_l3660_366010

/-- Given two vectors a and b in ℝ², where a = (2,1) and b is parallel to a with magnitude 2√5,
    prove that b must be either (4,2) or (-4,-2). -/
theorem parallel_vector_with_given_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  (∃ k : ℝ, b = (k * a.1, k * a.2)) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →
  b = (4, 2) ∨ b = (-4, -2) := by
  sorry

#check parallel_vector_with_given_magnitude

end NUMINAMATH_CALUDE_parallel_vector_with_given_magnitude_l3660_366010


namespace NUMINAMATH_CALUDE_octahedron_sum_l3660_366045

/-- A regular octahedron with numbers from 1 to 12 on its vertices -/
structure NumberedOctahedron where
  /-- The assignment of numbers to vertices -/
  vertex_numbers : Fin 6 → Fin 12
  /-- The property that each number from 1 to 12 is used exactly once -/
  all_numbers_used : Function.Injective vertex_numbers

/-- The sum of numbers on a face of the octahedron -/
def face_sum (o : NumberedOctahedron) (face : Fin 8) : ℕ := sorry

/-- The property that all face sums are equal -/
def all_face_sums_equal (o : NumberedOctahedron) : Prop :=
  ∀ (face1 face2 : Fin 8), face_sum o face1 = face_sum o face2

theorem octahedron_sum (o : NumberedOctahedron) (h : all_face_sums_equal o) :
  ∃ (face : Fin 8), face_sum o face = 39 := by sorry

end NUMINAMATH_CALUDE_octahedron_sum_l3660_366045


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_function_l3660_366049

theorem range_of_a_for_quadratic_function (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_function_l3660_366049


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3660_366094

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 16) : 
  min x y = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3660_366094


namespace NUMINAMATH_CALUDE_necklaces_remaining_l3660_366040

def initial_necklaces : ℕ := 360
def sold_percentage : ℚ := 45 / 100
def given_away_percentage : ℚ := 25 / 100

theorem necklaces_remaining (initial : ℕ) (sold_pct : ℚ) (given_pct : ℚ) : 
  initial = initial_necklaces →
  sold_pct = sold_percentage →
  given_pct = given_away_percentage →
  ⌊(initial - ⌊initial * sold_pct⌋) - ⌊(initial - ⌊initial * sold_pct⌋) * given_pct⌋⌋ = 149 :=
by sorry

end NUMINAMATH_CALUDE_necklaces_remaining_l3660_366040


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l3660_366032

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  2 * Real.sin α * Real.sin β * Real.cos γ = Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin γ ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l3660_366032


namespace NUMINAMATH_CALUDE_even_number_in_rows_l3660_366047

/-- Definition of the triangle table -/
def triangle_table : ℕ → ℤ → ℕ
| 1, 0 => 1
| n, k => if n > 1 ∧ abs k < n then
            triangle_table (n-1) (k-1) + triangle_table (n-1) k + triangle_table (n-1) (k+1)
          else 0

/-- Theorem: From the third row onward, each row contains at least one even number -/
theorem even_number_in_rows (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, Even (triangle_table n k) := by sorry

end NUMINAMATH_CALUDE_even_number_in_rows_l3660_366047


namespace NUMINAMATH_CALUDE_erin_curlers_count_l3660_366080

/-- Represents the number of curlers Erin put in her hair -/
def total_curlers : ℕ := 16

/-- Represents the number of small pink curlers -/
def pink_curlers : ℕ := total_curlers / 4

/-- Represents the number of medium blue curlers -/
def blue_curlers : ℕ := 2 * pink_curlers

/-- Represents the number of large green curlers -/
def green_curlers : ℕ := 4

/-- Proves that the total number of curlers is 16 -/
theorem erin_curlers_count :
  total_curlers = pink_curlers + blue_curlers + green_curlers :=
by sorry

end NUMINAMATH_CALUDE_erin_curlers_count_l3660_366080


namespace NUMINAMATH_CALUDE_ratio_determination_l3660_366063

/-- Given constants a and b, and unknowns x and y, the equation
    ax³ + bx²y + bxy² + ay³ = 0 can be transformed into a polynomial
    equation in terms of t, where t = x/y. -/
theorem ratio_determination (a b x y : ℝ) :
  ∃ t, t = x / y ∧ a * t^3 + b * t^2 + b * t + a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ratio_determination_l3660_366063


namespace NUMINAMATH_CALUDE_positive_numbers_properties_l3660_366058

theorem positive_numbers_properties (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_lt : a < b) (h_sum : a + b = 2) : 
  (1 < b ∧ b < 2) ∧ (Real.sqrt a + Real.sqrt b < 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_properties_l3660_366058


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3660_366027

theorem exponent_multiplication (a b : ℝ) : -a^2 * 2*a^4*b = -2*a^6*b := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3660_366027


namespace NUMINAMATH_CALUDE_expression_simplification_l3660_366065

theorem expression_simplification (x : ℝ) : 2*x - 3*(2 - x) + 4*(3 + x) - 5*(1 - 2*x) = 19*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3660_366065


namespace NUMINAMATH_CALUDE_complex_equality_l3660_366046

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3660_366046


namespace NUMINAMATH_CALUDE_certain_number_proof_l3660_366028

theorem certain_number_proof : ∃ x : ℝ, x * 2 + (12 + 4) * (1 / 8) = 602 ∧ x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3660_366028


namespace NUMINAMATH_CALUDE_division_problem_l3660_366018

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 1375) (h2 : quotient = 20) (h3 : remainder = 55) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 66 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3660_366018


namespace NUMINAMATH_CALUDE_phi_tau_ge_n_l3660_366089

/-- The number of divisors of a positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- Euler's totient function for a positive integer n -/
def phi (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, the product of φ(n) and τ(n) is greater than or equal to n -/
theorem phi_tau_ge_n (n : ℕ+) : phi n * tau n ≥ n := by sorry

end NUMINAMATH_CALUDE_phi_tau_ge_n_l3660_366089


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l3660_366077

/-- Rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The intersection area of two rectangles -/
def intersection_area (r1 r2 : Rectangle) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem intersection_area_theorem (abcd aecf : Rectangle) 
  (h1 : abcd.width = 11 ∧ abcd.height = 3)
  (h2 : aecf.width = 9 ∧ aecf.height = 7) :
  ∃ (m n : ℕ), 
    (intersection_area abcd aecf = m / n) ∧ 
    (are_relatively_prime m n) ∧ 
    (m + n = 109) := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l3660_366077


namespace NUMINAMATH_CALUDE_grandmother_age_multiple_l3660_366059

def milena_age : ℕ := 7

def grandfather_age_difference (grandmother_age : ℕ) : ℕ := grandmother_age + 2

theorem grandmother_age_multiple : ∃ (grandmother_age : ℕ), 
  grandfather_age_difference grandmother_age - milena_age = 58 ∧ 
  grandmother_age = 9 * milena_age := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_multiple_l3660_366059


namespace NUMINAMATH_CALUDE_cube_square_difference_property_l3660_366002

theorem cube_square_difference_property (x : ℝ) : 
  x^3 - x^2 = (x^2 - x)^2 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_square_difference_property_l3660_366002


namespace NUMINAMATH_CALUDE_janessa_cards_ordered_l3660_366003

/-- The number of cards Janessa ordered from eBay --/
def cards_ordered (initial_cards : ℕ) (father_cards : ℕ) (thrown_cards : ℕ) (given_cards : ℕ) (kept_cards : ℕ) : ℕ :=
  given_cards + kept_cards - (initial_cards + father_cards) + thrown_cards

theorem janessa_cards_ordered :
  cards_ordered 4 13 4 29 20 = 36 := by
  sorry

end NUMINAMATH_CALUDE_janessa_cards_ordered_l3660_366003


namespace NUMINAMATH_CALUDE_frog_corner_probability_l3660_366079

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up | Down | Left | Right | UpLeft | UpRight | DownLeft | DownRight

/-- The grid on which Frieda moves -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Calculates the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  sorry

/-- Calculates the probability of reaching a corner from a given position in n hops -/
def cornerProbability (grid : Grid) (p : Position) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of reaching any corner within 3 hops from (2,2) is 27/64 -/
theorem frog_corner_probability :
  let initialGrid : Grid := λ _ _ => 0
  let startPos : Position := ⟨1, 1⟩  -- (2,2) in 0-based indexing
  cornerProbability initialGrid startPos 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l3660_366079


namespace NUMINAMATH_CALUDE_spider_leg_count_l3660_366068

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

theorem spider_leg_count : total_legs = 112 := by
  sorry

end NUMINAMATH_CALUDE_spider_leg_count_l3660_366068
