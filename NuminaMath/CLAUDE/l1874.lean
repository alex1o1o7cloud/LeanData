import Mathlib

namespace bicycle_problem_l1874_187465

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 ∧ 
  speed_ratio = 1.2 ∧ 
  time_difference = 1/6 →
  ∃ (speed_B : ℝ),
    speed_B = 12 ∧
    distance / speed_B - time_difference = distance / (speed_ratio * speed_B) :=
by
  sorry

end bicycle_problem_l1874_187465


namespace exponent_sum_negative_one_l1874_187474

theorem exponent_sum_negative_one : (-1)^3 + (-1)^2 + (-1) = -1 := by
  sorry

end exponent_sum_negative_one_l1874_187474


namespace pen_sale_profit_percent_l1874_187413

/-- Calculates the profit percent for a pen sale scenario -/
theorem pen_sale_profit_percent 
  (num_pens : ℕ)
  (purchase_price : ℕ)
  (discount_percent : ℚ)
  (h1 : num_pens = 60)
  (h2 : purchase_price = 46)
  (h3 : discount_percent = 1/100) :
  ∃ (profit_percent : ℚ), abs (profit_percent - 2913/10000) < 1/10000 := by
  sorry

end pen_sale_profit_percent_l1874_187413


namespace successive_discounts_equivalence_l1874_187487

/-- Proves that a single discount of 40.5% is equivalent to two successive discounts of 15% and 30% on an item originally priced at $50. -/
theorem successive_discounts_equivalence :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.15
  let second_discount : ℝ := 0.30
  let equivalent_discount : ℝ := 0.405
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = original_price * (1 - equivalent_discount) := by
  sorry

end successive_discounts_equivalence_l1874_187487


namespace unique_solution_for_special_integers_l1874_187428

theorem unique_solution_for_special_integers (a b : ℕ+) : 
  a ≠ b → 
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, a + b^2 = p^k) → 
  (a + b^2 ∣ a^2 + b) → 
  a = 5 ∧ b = 2 := by
sorry

end unique_solution_for_special_integers_l1874_187428


namespace length_of_AB_l1874_187499

/-- Given a line segment AB with points P and Q on it, prove that AB has length 43.2 -/
theorem length_of_AB (A B P Q : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 3 / 8 →  -- P divides AB in ratio 3:5
  (Q.1 - A.1) / (B.1 - A.1) = 4 / 9 →  -- Q divides AB in ratio 4:5
  P.1 < Q.1 →  -- P and Q are on the same side of midpoint
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9 →  -- Distance between P and Q is 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 43.2^2 := by
sorry

end length_of_AB_l1874_187499


namespace triangle_segment_product_l1874_187436

/-- Given a triangle ABC with an interior point P, this theorem proves that
    if the segments created by extending lines from vertices through P
    to opposite sides have lengths a, b, c, and d, where a + b + c = 43
    and d = 3, then the product abc equals 441. -/
theorem triangle_segment_product (a b c d : ℝ) (h1 : a + b + c = 43) (h2 : d = 3) :
  a * b * c = 441 := by
  sorry

end triangle_segment_product_l1874_187436


namespace range_of_m_range_of_t_l1874_187457

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

def q (m : ℝ) : Prop := m^2 > 2*m + 8 ∧ 2*m + 8 > 0

def s (m t : ℝ) : Prop := t < m ∧ m < t + 1

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (p m ∧ q m) ↔ ((-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∀ m : ℝ, s m t → q m) ∧ (∃ m : ℝ, q m ∧ ¬s m t) ↔
  ((-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4) :=
sorry

end range_of_m_range_of_t_l1874_187457


namespace quadratic_bound_l1874_187471

theorem quadratic_bound (a b c : ℝ) (h1 : c > 0) (h2 : |a - b + c| ≤ 1) (h3 : |a + b + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ c + 1 / (4 * c) := by
  sorry

end quadratic_bound_l1874_187471


namespace circle_tangent_to_hyperbola_radius_l1874_187402

/-- The radius of a circle tangent to a hyperbola --/
theorem circle_tangent_to_hyperbola_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 5) : 
  let c := Real.sqrt (a^2 + b^2)
  let r := |a - c|
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧ 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) := by
  sorry

end circle_tangent_to_hyperbola_radius_l1874_187402


namespace gcd_fifteen_n_plus_five_nine_n_plus_four_l1874_187427

theorem gcd_fifteen_n_plus_five_nine_n_plus_four (n : ℕ) 
  (h_pos : n > 0) (h_mod : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 := by
  sorry

end gcd_fifteen_n_plus_five_nine_n_plus_four_l1874_187427


namespace cobalt_61_neutron_count_l1874_187421

/-- Represents an atom with its mass number and number of protons -/
structure Atom where
  mass_number : ℕ
  proton_count : ℕ

/-- Calculates the number of neutrons in an atom -/
def neutron_count (a : Atom) : ℕ := a.mass_number - a.proton_count

/-- Theorem: The number of neutrons in a ⁶¹₂₇Co atom is 34 -/
theorem cobalt_61_neutron_count :
  let co_61 : Atom := { mass_number := 61, proton_count := 27 }
  neutron_count co_61 = 34 := by
  sorry

end cobalt_61_neutron_count_l1874_187421


namespace fraction_cubes_equals_729_l1874_187416

theorem fraction_cubes_equals_729 : (81000 ^ 3) / (9000 ^ 3) = 729 := by
  sorry

end fraction_cubes_equals_729_l1874_187416


namespace trapezoid_AD_length_l1874_187411

/-- Represents a trapezoid ABCD with point M on CD and perpendicular AH from A to BM -/
structure Trapezoid :=
  (A B C D M H : ℝ × ℝ)
  (is_parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (M_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • C + t • D)
  (AH_perpendicular : (H.2 - A.2) * (M.1 - B.1) + (H.1 - A.1) * (M.2 - B.2) = 0)
  (AD_eq_HD : dist A D = dist H D)
  (BC_length : dist B C = 16)
  (CM_length : dist C M = 8)
  (MD_length : dist M D = 9)

/-- The length of AD in the given trapezoid is 18 -/
theorem trapezoid_AD_length (t : Trapezoid) : dist t.A t.D = 18 :=
  sorry

end trapezoid_AD_length_l1874_187411


namespace some_magical_not_spooky_l1874_187412

universe u

-- Define the types
variable {Creature : Type u}

-- Define the predicates
variable (Dragon : Creature → Prop)
variable (Magical : Creature → Prop)
variable (Spooky : Creature → Prop)

-- State the theorem
theorem some_magical_not_spooky
  (h1 : ∀ x, Dragon x → Magical x)
  (h2 : ∀ x, Spooky x → ¬ Dragon x) :
  ∃ x, Magical x ∧ ¬ Spooky x :=
sorry

end some_magical_not_spooky_l1874_187412


namespace coefficient_x4_in_product_l1874_187431

/-- The coefficient of x^4 in the product of two specific polynomials -/
theorem coefficient_x4_in_product : 
  let p1 : Polynomial ℚ := X^5 - 4*X^4 + 6*X^3 - 7*X^2 + 2*X - 1
  let p2 : Polynomial ℚ := 3*X^4 - 2*X^3 + 5*X - 8
  (p1 * p2).coeff 4 = 27 := by sorry

end coefficient_x4_in_product_l1874_187431


namespace base3_to_base9_first_digit_l1874_187494

def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base9 (n : Nat) : Nat :=
  Nat.log 9 n + 1

theorem base3_to_base9_first_digit :
  let x : Nat := base3_to_decimal [1,2,1,1,2,2,1,1,1,2,2,2,1,1,1,1,2,2,2,2]
  first_digit_base9 x = 5 := by
  sorry

end base3_to_base9_first_digit_l1874_187494


namespace a_range_l1874_187491

-- Define proposition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

-- Define proposition q
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the theorem
theorem a_range (a : ℝ) : 
  (a < 0) → 
  (∀ x, ¬(p x a) → ¬(q x)) → 
  (∃ x, ¬(p x a) ∧ (q x)) → 
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
sorry

end a_range_l1874_187491


namespace locus_is_ellipse_and_angle_bisector_l1874_187403

-- Define the circle A
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus of points N
def locus_N (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ),
    circle_A mx my ∧
    (x - mx)^2 + (y - my)^2 = (x - point_B.1)^2 + (y - point_B.2)^2 ∧
    (x - mx) * (1 - mx) + (y - my) * (0 - my) = 0

-- Theorem statement
theorem locus_is_ellipse_and_angle_bisector :
  (∀ x y, locus_N x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ k x₁ y₁ x₂ y₂,
    x₁^2/4 + y₁^2/3 = 1 ∧
    x₂^2/4 + y₂^2/3 = 1 ∧
    y₁ = k*(x₁ - 1) ∧
    y₂ = k*(x₂ - 1) →
    (y₁ / (x₁ - 4) + y₂ / (x₂ - 4) = 0)) :=
sorry

end locus_is_ellipse_and_angle_bisector_l1874_187403


namespace product_equals_72_17_l1874_187455

/-- Represents the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal 0.456̄ and 9 -/
def product : ℚ := 9 * repeating_decimal

theorem product_equals_72_17 : product = 72 / 17 := by sorry

end product_equals_72_17_l1874_187455


namespace work_completion_l1874_187406

/-- The number of days A and B worked together before A left the job -/
def days_worked_together : ℕ := 5

/-- A's work rate in terms of the total work per day -/
def rate_A : ℚ := 1 / 20

/-- B's work rate in terms of the total work per day -/
def rate_B : ℚ := 1 / 12

/-- The number of days B worked alone after A left -/
def days_B_alone : ℕ := 3

theorem work_completion :
  (days_worked_together : ℚ) * (rate_A + rate_B) + (days_B_alone : ℚ) * rate_B = 1 := by
  sorry

#check work_completion

end work_completion_l1874_187406


namespace sanctuary_feeding_sequences_l1874_187480

/-- Represents the number of pairs of animals in the sanctuary -/
def num_pairs : ℕ := 5

/-- Calculates the number of distinct feeding sequences for animals in a sanctuary -/
def feeding_sequences (n : ℕ) : ℕ :=
  let male_choices := List.range n
  let female_choices := List.range n
  (female_choices.foldl (· * ·) 1) * (male_choices.tail.foldl (· * ·) 1)

/-- Theorem stating the number of distinct feeding sequences for the given conditions -/
theorem sanctuary_feeding_sequences :
  feeding_sequences num_pairs = 5760 :=
sorry

end sanctuary_feeding_sequences_l1874_187480


namespace sphere_radius_from_surface_area_l1874_187479

theorem sphere_radius_from_surface_area (surface_area : Real) (radius : Real) : 
  surface_area = 64 * Real.pi → 4 * Real.pi * radius^2 = surface_area → radius = 4 := by
  sorry

end sphere_radius_from_surface_area_l1874_187479


namespace chinese_remainder_theorem_l1874_187450

theorem chinese_remainder_theorem (y : ℤ) : 
  (y + 4 ≡ 3^2 [ZMOD 3^3]) → 
  (y + 4 ≡ 4^2 [ZMOD 5^3]) → 
  (y + 4 ≡ 6^2 [ZMOD 7^3]) → 
  (y ≡ 32 [ZMOD 105]) := by
  sorry

end chinese_remainder_theorem_l1874_187450


namespace quadratic_trinomial_equality_l1874_187453

/-- A quadratic trinomial function -/
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_equality (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, quadratic_trinomial a b c (3.8 * x - 1) = quadratic_trinomial a b c (-3.8 * x)) →
  b = a := by sorry

end quadratic_trinomial_equality_l1874_187453


namespace taps_teps_equivalence_l1874_187401

/-- Given the equivalences between taps, tops, and teps, prove that 15 taps are equivalent to 48 teps -/
theorem taps_teps_equivalence (tap top tep : ℕ → ℚ) 
  (h1 : 5 * tap 1 = 4 * top 1)  -- 5 taps are equivalent to 4 tops
  (h2 : 3 * top 1 = 12 * tep 1) -- 3 tops are equivalent to 12 teps
  : 15 * tap 1 = 48 * tep 1 := by
  sorry

end taps_teps_equivalence_l1874_187401


namespace union_of_P_and_complement_of_Q_l1874_187449

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem union_of_P_and_complement_of_Q :
  P ∪ (univ \ Q) = Iic (-2) ∪ Ici 1 := by sorry

end union_of_P_and_complement_of_Q_l1874_187449


namespace first_group_size_l1874_187483

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The daily wage in rupees -/
def daily_wage : ℚ := sorry

theorem first_group_size :
  (M * 10 * daily_wage = 1200) ∧
  (9 * 6 * daily_wage = 1620) →
  M = 4 := by
  sorry

end first_group_size_l1874_187483


namespace reflection_F_to_H_l1874_187441

/-- Represents the possible shapes in the problem -/
inductive Shape
  | F
  | E
  | H
  | Other

/-- Represents the types of reflections -/
inductive Reflection
  | Vertical
  | Horizontal

/-- Applies a reflection to a shape -/
def applyReflection (s : Shape) (r : Reflection) : Shape :=
  match s, r with
  | Shape.F, Reflection.Vertical => Shape.E
  | Shape.E, Reflection.Horizontal => Shape.H
  | _, _ => Shape.Other

/-- Theorem stating that applying vertical then horizontal reflection to F results in H -/
theorem reflection_F_to_H :
  applyReflection (applyReflection Shape.F Reflection.Vertical) Reflection.Horizontal = Shape.H :=
by sorry

end reflection_F_to_H_l1874_187441


namespace q_value_approximation_l1874_187486

theorem q_value_approximation (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p*q = 16/3) : 
  ∃ ε > 0, |q - 7.27| < ε :=
sorry

end q_value_approximation_l1874_187486


namespace sum_of_digits_next_l1874_187429

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem sum_of_digits_next (n : ℕ) : S n = 1384 → S (n + 1) = 15 := by
  sorry

end sum_of_digits_next_l1874_187429


namespace total_wheels_count_l1874_187467

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of wheels on a unicycle -/
def unicycle_wheels : ℕ := 1

/-- The number of wheels on a four-wheeled scooter -/
def scooter_wheels : ℕ := 4

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The number of teenagers riding unicycles -/
def teenagers_on_unicycles : ℕ := 3

/-- The number of children riding four-wheeled scooters -/
def children_on_scooters : ℕ := 8

/-- The total number of wheels Dimitri saw at the park -/
def total_wheels : ℕ := 
  adults_on_bicycles * bicycle_wheels +
  children_on_tricycles * tricycle_wheels +
  teenagers_on_unicycles * unicycle_wheels +
  children_on_scooters * scooter_wheels

theorem total_wheels_count : total_wheels = 92 := by
  sorry

end total_wheels_count_l1874_187467


namespace father_current_age_l1874_187498

/-- The father's current age -/
def father_age : ℕ := sorry

/-- The son's current age -/
def son_age : ℕ := sorry

/-- Six years ago, the father's age was five times the son's age -/
axiom past_age_relation : father_age - 6 = 5 * (son_age - 6)

/-- In six years, the sum of their ages will be 78 -/
axiom future_age_sum : father_age + 6 + son_age + 6 = 78

theorem father_current_age : father_age = 51 := by
  sorry

end father_current_age_l1874_187498


namespace word_sum_equation_l1874_187414

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the function that represents the word "TWENTY"
def twenty (t w e n y : Digit) : ℕ :=
  10 * (10 * t.val + w.val) + e.val * 10 + n.val * 1 + y.val

-- Define the function that represents the word "TEN"
def ten (t e n : Digit) : ℕ :=
  10 * t.val + e.val * 1 + n.val

-- Main theorem
theorem word_sum_equation :
  ∃! (e g h i n t w y : Digit),
    twenty t w e n y + twenty t w e n y + twenty t w e n y + ten t e n + ten t e n = 80 ∧
    e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ n ∧ e ≠ t ∧ e ≠ w ∧ e ≠ y ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ n ∧ g ≠ t ∧ g ≠ w ∧ g ≠ y ∧
    h ≠ i ∧ h ≠ n ∧ h ≠ t ∧ h ≠ w ∧ h ≠ y ∧
    i ≠ n ∧ i ≠ t ∧ i ≠ w ∧ i ≠ y ∧
    n ≠ t ∧ n ≠ w ∧ n ≠ y ∧
    t ≠ w ∧ t ≠ y ∧
    w ≠ y := by
  sorry

end word_sum_equation_l1874_187414


namespace hyperbola_focal_length_l1874_187481

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 2 * x + m * y = 0

-- Theorem statement
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  (∃ x y : ℝ, hyperbola m x y ∧ asymptote m x y) →
  ∃ a b : ℝ, a^2 = m ∧ b^2 = 1 ∧ 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3 :=
sorry

end hyperbola_focal_length_l1874_187481


namespace parabola_tangent_to_line_l1874_187447

/-- A parabola y = ax^2 + 12 is tangent to the line y = 2x if and only if a = 1/12 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 12 = 2 * x) ↔ a = 1/12 := by
  sorry

end parabola_tangent_to_line_l1874_187447


namespace inequality_proof_l1874_187456

theorem inequality_proof (A B C : ℝ) (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C) 
  (h_ineq : A^4 + B^4 + C^4 ≤ 2*(A^2*B^2 + B^2*C^2 + C^2*A^2)) :
  A^2 + B^2 + C^2 ≤ 2*(A*B + B*C + C*A) := by
  sorry

end inequality_proof_l1874_187456


namespace ellipse_properties_l1874_187444

/-- Ellipse C with given properties -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (2^2 / a^2) + (2^2 / b^2) = 1)
  (h4 : a^2 - b^2 = 2 * b^2)

/-- Line l passing through (-1, 0) -/
def Line := ℝ → ℝ

/-- Intersection points of line l and ellipse C -/
def IntersectionPoints (C : Ellipse) (l : Line) := ℝ × ℝ

/-- Foci of ellipse C -/
def Foci (C : Ellipse) := ℝ × ℝ

/-- Areas of triangles formed by foci and intersection points -/
def TriangleAreas (C : Ellipse) (l : Line) := ℝ × ℝ

/-- Main theorem -/
theorem ellipse_properties (C : Ellipse) (l : Line) :
  (∀ x y : ℝ, x^2 / 12 + y^2 / 6 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∃ S : Set ℝ, S = {x | 0 ≤ x ∧ x ≤ Real.sqrt 3} ∧
    ∀ areas : TriangleAreas C l, |areas.1 - areas.2| ∈ S) :=
sorry

end ellipse_properties_l1874_187444


namespace product_from_gcd_lcm_l1874_187404

theorem product_from_gcd_lcm (a b : ℕ+) :
  Nat.gcd a b = 5 → Nat.lcm a b = 60 → a * b = 300 := by
  sorry

end product_from_gcd_lcm_l1874_187404


namespace proposition_p_true_q_false_l1874_187437

theorem proposition_p_true_q_false :
  (∃ x : ℝ, Real.exp x ≥ x + 1) ∧
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) :=
by sorry

end proposition_p_true_q_false_l1874_187437


namespace complex_magnitude_l1874_187469

theorem complex_magnitude (z : ℂ) : z = Complex.I * (2 - Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1874_187469


namespace girls_count_l1874_187432

def total_students : ℕ := 8

def probability_2boys_1girl : ℚ := 15/28

theorem girls_count (x : ℕ) 
  (h1 : x ≤ total_students)
  (h2 : Nat.choose (total_students - x) 2 * Nat.choose x 1 / Nat.choose total_students 3 = probability_2boys_1girl) :
  x = 2 ∨ x = 3 := by
  sorry

end girls_count_l1874_187432


namespace three_divisors_of_2469_minus_5_l1874_187496

theorem three_divisors_of_2469_minus_5 : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (2469 : ℤ).natAbs % (m^2 - 5 : ℤ).natAbs = 0) ∧ 
    S.card = 3 := by
  sorry

end three_divisors_of_2469_minus_5_l1874_187496


namespace earnings_difference_l1874_187438

def saheed_earnings : ℕ := 216
def vika_earnings : ℕ := 84

def kayla_earnings : ℕ := saheed_earnings / 4

theorem earnings_difference : vika_earnings - kayla_earnings = 30 :=
by sorry

end earnings_difference_l1874_187438


namespace tens_digit_of_8_pow_2048_l1874_187490

theorem tens_digit_of_8_pow_2048 : 8^2048 % 100 = 88 := by sorry

end tens_digit_of_8_pow_2048_l1874_187490


namespace min_value_expression_l1874_187489

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 := by
  sorry

end min_value_expression_l1874_187489


namespace dara_employment_wait_l1874_187492

/-- The minimum age required for employment at the company -/
def min_age : ℕ := 25

/-- Jane's current age -/
def jane_age : ℕ := 28

/-- The number of years in the future when Dara will be half Jane's age -/
def future_years : ℕ := 6

/-- Calculates Dara's current age based on the given conditions -/
def dara_current_age : ℕ :=
  (jane_age + future_years) / 2 - future_years

/-- The time before Dara reaches the minimum age for employment -/
def time_to_min_age : ℕ := min_age - dara_current_age

/-- Theorem stating that the time before Dara reaches the minimum age for employment is 14 years -/
theorem dara_employment_wait : time_to_min_age = 14 := by
  sorry

end dara_employment_wait_l1874_187492


namespace bus_children_count_l1874_187461

/-- The number of children on a bus before and after a bus stop. -/
theorem bus_children_count (after : ℕ) (difference : ℕ) (before : ℕ) 
  (h1 : after = 18)
  (h2 : difference = 23)
  (h3 : before = after + difference) :
  before = 41 := by
  sorry

end bus_children_count_l1874_187461


namespace area_of_region_R_l1874_187478

/-- A regular hexagon with side length 3 -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = 3
  -- Additional properties of regularity could be added here

/-- The region R in the hexagon -/
def region_R (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {p | p ∈ interior h ∧ 
       ∀ i : Fin 6, i ≠ 0 → dist p (h.vertices 0) < dist p (h.vertices i)}
  where
  interior : RegularHexagon → Set (ℝ × ℝ) := sorry  -- Definition of hexagon interior

/-- The area of a set in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

theorem area_of_region_R (h : RegularHexagon) : 
  area (region_R h) = 27 * Real.sqrt 3 / 16 := by sorry

end area_of_region_R_l1874_187478


namespace square_of_1023_l1874_187460

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  -- Proof goes here
  sorry

end square_of_1023_l1874_187460


namespace smallest_balloon_count_l1874_187446

def balloon_count (color : String) : ℕ :=
  match color with
  | "red" => 10
  | "blue" => 8
  | "yellow" => 5
  | "green" => 6
  | _ => 0

def has_seven_or_more (color : String) : Bool :=
  balloon_count color ≥ 7

def colors_with_seven_or_more : List String :=
  ["red", "blue", "yellow", "green"].filter has_seven_or_more

theorem smallest_balloon_count : 
  (colors_with_seven_or_more.map balloon_count).minimum? = some 8 := by
  sorry

end smallest_balloon_count_l1874_187446


namespace parallel_vectors_imply_a_value_l1874_187407

-- Define the vectors m and n
def m : ℝ × ℝ := (-2, 3)
def n (a : ℝ) : ℝ × ℝ := (a + 1, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_imply_a_value :
  parallel m (n a) → a = -3 :=
by sorry

end parallel_vectors_imply_a_value_l1874_187407


namespace geometric_sequence_sum_l1874_187419

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 6 = 10 →
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 100 := by
  sorry

end geometric_sequence_sum_l1874_187419


namespace salt_solution_mixture_l1874_187442

/-- Given a mixture of pure water and salt solution, find the volume of salt solution needed. -/
theorem salt_solution_mixture (x : ℝ) : 
  (0.15 * (1 + x) = 0.45 * x) → x = 0.5 := by
  sorry

end salt_solution_mixture_l1874_187442


namespace equal_volumes_of_modified_cylinders_l1874_187452

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders :
  let r : ℝ := 5  -- Initial radius in inches
  let h : ℝ := 4  -- Initial height in inches
  let dr : ℝ := 2  -- Increase in radius for the first cylinder
  ∀ y : ℝ,  -- Increase in height for the second cylinder
  y ≠ 0 →  -- y is non-zero
  π * (r + dr)^2 * h = π * r^2 * (h + y) →  -- Volumes are equal
  y = 96 / 25 := by
sorry

end equal_volumes_of_modified_cylinders_l1874_187452


namespace combined_ppf_theorem_combined_ppf_range_l1874_187495

/-- Production Possibility Frontier (PPF) for a single female -/
def single_ppf (K : ℝ) : ℝ := 40 - 2 * K

/-- Combined Production Possibility Frontier (PPF) for two females -/
def combined_ppf (K : ℝ) : ℝ := 80 - 2 * K

/-- Theorem stating that the combined PPF of two identical linear PPFs is the sum of their individual PPFs -/
theorem combined_ppf_theorem (K : ℝ) (h : K ≤ 40) :
  combined_ppf K = single_ppf (K / 2) + single_ppf (K / 2) :=
by sorry

/-- Corollary stating the range of K for the combined PPF -/
theorem combined_ppf_range (K : ℝ) :
  K ≤ 40 ↔ ∃ (K1 K2 : ℝ), K1 ≤ 20 ∧ K2 ≤ 20 ∧ K = K1 + K2 :=
by sorry

end combined_ppf_theorem_combined_ppf_range_l1874_187495


namespace parallel_line_plane_m_value_l1874_187475

/-- Given a line with direction vector (2, m, 1) parallel to a plane with normal vector (1, 1/2, 2), prove m = -8 -/
theorem parallel_line_plane_m_value (m : ℝ) : 
  let vec_m : Fin 3 → ℝ := ![2, m, 1]
  let vec_n : Fin 3 → ℝ := ![1, 1/2, 2]
  (∀ i : Fin 3, vec_m i * vec_n i = 0) → m = -8 := by
  sorry


end parallel_line_plane_m_value_l1874_187475


namespace lauras_weekly_driving_distance_l1874_187482

/-- Laura's weekly driving distance calculation -/
theorem lauras_weekly_driving_distance :
  -- Definitions based on given conditions
  let round_trip_to_school : ℕ := 20
  let supermarket_distance_from_school : ℕ := 10
  let days_to_school_per_week : ℕ := 7
  let supermarket_trips_per_week : ℕ := 2

  -- Derived calculations
  let round_trip_to_supermarket : ℕ := round_trip_to_school + 2 * supermarket_distance_from_school
  let weekly_school_distance : ℕ := days_to_school_per_week * round_trip_to_school
  let weekly_supermarket_distance : ℕ := supermarket_trips_per_week * round_trip_to_supermarket

  -- Theorem statement
  weekly_school_distance + weekly_supermarket_distance = 220 := by
  sorry

end lauras_weekly_driving_distance_l1874_187482


namespace systematic_sample_fourth_element_l1874_187466

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sampleSize : ℕ
  interval : ℕ
  startPoint : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.startPoint + k * s.interval ∧ k < s.sampleSize

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_interval : s.interval = s.total / s.sampleSize)
  (h_start : s.startPoint = 6)
  (h_contains_6 : s.contains 6)
  (h_contains_32 : s.contains 32)
  (h_contains_45 : s.contains 45) :
  s.contains 19 :=
sorry

end systematic_sample_fourth_element_l1874_187466


namespace hemisphere_surface_area_l1874_187443

theorem hemisphere_surface_area :
  let sphere_surface_area (r : ℝ) := 4 * Real.pi * r^2
  let base_area := 3
  let hemisphere_surface_area (r : ℝ) := 2 * Real.pi * r^2 + base_area
  ∃ r : ℝ, base_area = Real.pi * r^2 ∧ hemisphere_surface_area r = 9 :=
by sorry

end hemisphere_surface_area_l1874_187443


namespace volleyball_team_physics_count_l1874_187448

theorem volleyball_team_physics_count 
  (total_players : ℕ) 
  (math_players : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_players = 15)
  (h2 : math_players = 10)
  (h3 : both_subjects = 4)
  (h4 : both_subjects ≤ math_players)
  (h5 : math_players ≤ total_players) :
  total_players - (math_players - both_subjects) = 9 :=
by sorry

end volleyball_team_physics_count_l1874_187448


namespace angle_R_measure_l1874_187462

/-- A hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Angle F in degrees
  I : ℝ  -- Angle I in degrees
  G : ℝ  -- Angle G in degrees
  U : ℝ  -- Angle U in degrees
  R : ℝ  -- Angle R in degrees
  E : ℝ  -- Angle E in degrees
  sum_angles : F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  supplementary : G + U = 180  -- G and U are supplementary
  congruent : F = I ∧ I = R ∧ R = E  -- F, I, R, and E are congruent

/-- The measure of angle R in the specific hexagon is 135 degrees -/
theorem angle_R_measure (h : Hexagon) : h.R = 135 := by
  sorry

end angle_R_measure_l1874_187462


namespace biased_coin_probabilities_l1874_187435

theorem biased_coin_probabilities (p : ℝ) 
  (h_range : 0 < p ∧ p < 1)
  (h_equal_prob : (5 : ℝ) * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3)
  (h_non_zero : (5 : ℝ) * p * (1 - p)^4 ≠ 0) : 
  p = 1/3 ∧ (10 : ℝ) * p^3 * (1 - p)^2 = 40/243 := by
  sorry

end biased_coin_probabilities_l1874_187435


namespace symmetry_axis_condition_l1874_187477

theorem symmetry_axis_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) ↔ x = (p * (-y) + q) / (r * (-y) + s)) →
  p + s = 0 := by
  sorry

end symmetry_axis_condition_l1874_187477


namespace multiplication_puzzle_l1874_187422

theorem multiplication_puzzle (c d : ℕ) : 
  c ≤ 9 → d ≤ 9 → (30 + c) * (10 * d + 4) = 132 → c + d = 11 := by
  sorry

end multiplication_puzzle_l1874_187422


namespace impossibleTiling_l1874_187409

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Function to check if a modified chessboard can be tiled with dominoes -/
def canTileChessboard (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.size = 8 ∧
  board.cornersRemoved = 2 ∧
  tile.length = 2 ∧
  tile.width = 1 ∧
  ∃ (tiling : Nat), False  -- This represents the impossibility of tiling

/-- Theorem stating that it's impossible to tile the modified chessboard with 2x1 dominoes -/
theorem impossibleTiling (board : ModifiedChessboard) (tile : Domino) :
  ¬(canTileChessboard board tile) :=
by
  sorry

end impossibleTiling_l1874_187409


namespace decimal_to_fraction_l1874_187468

theorem decimal_to_fraction (x : ℚ) (h : x = 368/100) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ (Nat.gcd n d = 1) :=
by sorry

end decimal_to_fraction_l1874_187468


namespace polygon_diagonals_l1874_187408

theorem polygon_diagonals (n : ℕ) (h : n > 0) : 
  3 * n = n * (n - 3) / 2 ↔ n = 9 := by
  sorry

end polygon_diagonals_l1874_187408


namespace exp_inequality_l1874_187400

theorem exp_inequality (a b : ℝ) (h : a > b) : Real.exp (-a) - Real.exp (-b) < 0 := by
  sorry

end exp_inequality_l1874_187400


namespace guaranteed_scores_l1874_187415

/-- Represents a player in the card game -/
inductive Player : Type
| One
| Two

/-- Represents a card in the game -/
def Card := Nat

/-- The deck of cards for Player One -/
def player_one_deck : List Card := List.range 1000 |>.map (fun n => 2 * n + 2)

/-- The deck of cards for Player Two -/
def player_two_deck : List Card := List.range 1001 |>.map (fun n => 2 * n + 1)

/-- The number of rounds in the game -/
def num_rounds : Nat := 1000

/-- A strategy for playing the game -/
def Strategy := List Card → List Card → Card

/-- The result of playing the game -/
structure GameResult where
  player_one_score : Nat
  player_two_score : Nat

/-- Play the game with given strategies -/
def play_game (strategy_one strategy_two : Strategy) : GameResult :=
  sorry

/-- Theorem stating the guaranteed minimum scores for both players -/
theorem guaranteed_scores :
  (∃ (strategy_one : Strategy),
    ∀ (strategy_two : Strategy),
      (play_game strategy_one strategy_two).player_one_score ≥ 499) ∧
  (∃ (strategy_two : Strategy),
    ∀ (strategy_one : Strategy),
      (play_game strategy_one strategy_two).player_two_score ≥ 501) :=
sorry

end guaranteed_scores_l1874_187415


namespace point_in_first_quadrant_l1874_187434

def complex_i : ℂ := Complex.I

theorem point_in_first_quadrant (z : ℂ) :
  z = complex_i * (2 - 3 * complex_i) →
  Complex.re z > 0 ∧ Complex.im z > 0 :=
by sorry

end point_in_first_quadrant_l1874_187434


namespace marys_oranges_l1874_187451

theorem marys_oranges (jason_oranges total_oranges : ℕ) 
  (h1 : jason_oranges = 105)
  (h2 : total_oranges = 227)
  : total_oranges - jason_oranges = 122 := by
  sorry

end marys_oranges_l1874_187451


namespace set_relationship_l1874_187425

-- Define the sets P, Q, and S
def P : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ Real.log (x * y) = Real.log x + Real.log y}
def Q : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ (2 : ℝ)^x * (2 : ℝ)^y = (2 : ℝ)^(x + y)}
def S : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ Real.sqrt x * Real.sqrt y = Real.sqrt (x * y)}

-- State the theorem
theorem set_relationship : P ⊆ S ∧ S ⊆ Q := by sorry

end set_relationship_l1874_187425


namespace min_value_expression_l1874_187445

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a = 2 * b) (hac : a = 2 * c) :
  (a + b) / c + (a + c) / b + (b + c) / a = 9.25 := by
  sorry

end min_value_expression_l1874_187445


namespace divisibility_implication_l1874_187433

theorem divisibility_implication (a : ℤ) : 
  (8 ∣ (5*a + 3) * (3*a + 1)) → (16 ∣ (5*a + 3) * (3*a + 1)) := by
  sorry

end divisibility_implication_l1874_187433


namespace perfect_square_condition_l1874_187420

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, 4^27 + 4^1000 + 4^(n:ℕ) = m^2) → n ≤ 1972 := by
sorry

end perfect_square_condition_l1874_187420


namespace pie_eating_contest_l1874_187458

theorem pie_eating_contest (bill_pies adam_pies sierra_pies : ℕ) : 
  adam_pies = bill_pies + 3 →
  sierra_pies = 2 * bill_pies →
  bill_pies + adam_pies + sierra_pies = 27 →
  sierra_pies = 12 := by
sorry

end pie_eating_contest_l1874_187458


namespace hcf_of_product_and_greater_l1874_187493

theorem hcf_of_product_and_greater (a b : ℕ) (h1 : a * b = 4107) (h2 : a = 111) :
  Nat.gcd a b = 37 := by
  sorry

end hcf_of_product_and_greater_l1874_187493


namespace rectangular_prism_equal_surface_volume_l1874_187430

theorem rectangular_prism_equal_surface_volume (a b c : ℕ) :
  (2 * (a * b + b * c + c * a) = a * b * c) ∧ (c = a * b / 2) →
  ((a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12)) := by
  sorry

end rectangular_prism_equal_surface_volume_l1874_187430


namespace wheel_rotation_coincidence_l1874_187439

/-- The distance the larger wheel must roll for two initially coincident points to coincide again -/
theorem wheel_rotation_coincidence (big_circ small_circ : ℕ) 
  (h_big : big_circ = 12) 
  (h_small : small_circ = 8) : 
  Nat.lcm big_circ small_circ = 24 := by
  sorry

end wheel_rotation_coincidence_l1874_187439


namespace min_toothpicks_to_remove_for_given_figure_l1874_187485

/-- Represents a figure made of toothpicks forming squares and triangles. -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  num_squares : ℕ
  num_triangles : ℕ
  toothpicks_per_square : ℕ
  toothpicks_per_triangle : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all shapes. -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove for the given figure. -/
theorem min_toothpicks_to_remove_for_given_figure :
  let figure : ToothpickFigure := {
    total_toothpicks := 40,
    num_squares := 10,
    num_triangles := 15,
    toothpicks_per_square := 4,
    toothpicks_per_triangle := 3
  }
  min_toothpicks_to_remove figure = 10 := by sorry

end min_toothpicks_to_remove_for_given_figure_l1874_187485


namespace parabola_vertex_x_coordinate_l1874_187426

/-- Given a quadratic function f(x) = ax² + bx + c passing through
    (-2,5), (4,5), and (2,2), prove that the x-coordinate of its vertex is 1. -/
theorem parabola_vertex_x_coordinate 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f (-2) = 5) 
  (h3 : f 4 = 5) 
  (h4 : f 2 = 2) : 
  (- b) / (2 * a) = 1 := by sorry

end parabola_vertex_x_coordinate_l1874_187426


namespace javier_donut_fundraising_l1874_187417

/-- Represents the problem of calculating Javier's fundraising for a new game through donut sales. -/
theorem javier_donut_fundraising
  (dozen_cost : ℚ)
  (donut_price : ℚ)
  (donuts_per_dozen : ℕ)
  (dozens_to_sell : ℕ)
  (h1 : dozen_cost = 240 / 100)  -- $2.40 per dozen
  (h2 : donut_price = 1)         -- $1 per donut
  (h3 : donuts_per_dozen = 12)   -- 12 donuts in a dozen
  (h4 : dozens_to_sell = 10)     -- Selling 10 dozens
  : (dozens_to_sell * donuts_per_dozen * donut_price) - (dozens_to_sell * dozen_cost) = 96 := by
  sorry


end javier_donut_fundraising_l1874_187417


namespace sin_cos_difference_l1874_187454

theorem sin_cos_difference (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = 1/5) → 
  (Real.sin α - Real.cos α = 7/5) := by
sorry

end sin_cos_difference_l1874_187454


namespace reflection_theorem_l1874_187410

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a reflection plane -/
inductive ReflectionPlane
  | XY
  | YZ
  | ZX

/-- Reflects a vector across a given plane -/
def reflect (v : Vector3D) (plane : ReflectionPlane) : Vector3D :=
  match plane with
  | ReflectionPlane.XY => ⟨v.x, v.y, -v.z⟩
  | ReflectionPlane.YZ => ⟨-v.x, v.y, v.z⟩
  | ReflectionPlane.ZX => ⟨v.x, -v.y, v.z⟩

/-- Reflects a vector across all three coordinate planes -/
def reflectAll (v : Vector3D) : Vector3D :=
  reflect (reflect (reflect v ReflectionPlane.XY) ReflectionPlane.YZ) ReflectionPlane.ZX

theorem reflection_theorem (v : Vector3D) :
  reflectAll v = Vector3D.mk (-v.x) (-v.y) (-v.z) := by
  sorry

#check reflection_theorem

end reflection_theorem_l1874_187410


namespace quadratic_greatest_lower_bound_l1874_187464

/-- The greatest lower bound of a quadratic function -/
theorem quadratic_greatest_lower_bound (a b : ℝ) (ha : a ≠ 0) (hnz : a ≠ 0 ∨ b ≠ 0) (hpos : a > 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x
  ∃ M : ℝ, M = -b^2 / (4 * a) ∧ ∀ x, f x ≥ M ∧ ∀ N, (∀ x, f x ≥ N) → N ≤ M :=
by sorry

end quadratic_greatest_lower_bound_l1874_187464


namespace trisection_distances_l1874_187405

/-- An isosceles triangle with given distances to trisection points -/
structure IsoTriangle where
  -- Side lengths
  ab : ℝ
  ac : ℝ
  bc : ℝ
  -- Distances from C to trisection points of AB
  d1 : ℝ
  d2 : ℝ
  -- Triangle is isosceles
  isIsosceles : ab = ac
  -- d1 and d2 are the given distances
  distancesGiven : (d1 = 17 ∧ d2 = 20) ∨ (d1 = 20 ∧ d2 = 17)

/-- The theorem to be proved -/
theorem trisection_distances (t : IsoTriangle) :
  let x := Real.sqrt ((8 * t.d2^2 + 5 * t.d1^2) / 3)
  x = Real.sqrt 585 ∨ x = Real.sqrt 104 := by
  sorry

end trisection_distances_l1874_187405


namespace A_power_50_l1874_187418

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; -4, -1]

theorem A_power_50 : 
  A^50 = 50 * 8^49 • A - 399 * 8^49 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end A_power_50_l1874_187418


namespace prism_volume_and_ak_length_l1874_187440

/-- Regular triangular prism with inscribed sphere -/
structure RegularPrismWithSphere where
  -- Height of the prism
  h : ℝ
  -- Radius of the inscribed sphere
  r : ℝ
  -- Point K on edge AA₁
  k : ℝ
  -- Point L on edge BB₁
  l : ℝ
  -- Assumption that h = 12
  h_eq : h = 12
  -- Assumption that r = √(35/3)
  r_eq : r = Real.sqrt (35/3)
  -- Assumption that KL is parallel to AB
  kl_parallel_ab : True  -- We can't directly express this geometric condition
  -- Assumption that plane KBC touches the sphere
  kbc_touches_sphere : True  -- We can't directly express this geometric condition
  -- Assumption that plane LA₁C₁ touches the sphere
  la1c1_touches_sphere : True  -- We can't directly express this geometric condition

/-- Theorem about the volume and AK length of the regular triangular prism with inscribed sphere -/
theorem prism_volume_and_ak_length (p : RegularPrismWithSphere) :
  ∃ (v : ℝ) (ak : ℝ),
    v = 420 * Real.sqrt 3 ∧
    (ak = 8 ∨ ak = 4) :=
by sorry

end prism_volume_and_ak_length_l1874_187440


namespace common_difference_of_arithmetic_sequence_l1874_187423

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ + d * (i : ℝ)

theorem common_difference_of_arithmetic_sequence
  (a₁ : ℝ) (aₙ : ℝ) (S : ℝ) (n : ℕ) (d : ℝ)
  (h₁ : a₁ = 5)
  (h₂ : aₙ = 50)
  (h₃ : S = 275)
  (h₄ : aₙ = a₁ + d * (n - 1))
  (h₅ : S = n / 2 * (a₁ + aₙ))
  : d = 5 := by
  sorry

#check common_difference_of_arithmetic_sequence

end common_difference_of_arithmetic_sequence_l1874_187423


namespace factorial_sum_equality_l1874_187424

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 3 = 35904 := by
  sorry

end factorial_sum_equality_l1874_187424


namespace equation_equivalence_implies_mnp_18_l1874_187470

theorem equation_equivalence_implies_mnp_18 
  (a x z c : ℝ) (m n p : ℤ) 
  (h : a^8*x*z - a^7*z - a^6*x = a^5*(c^5 - 1)) 
  (h_equiv : (a^m*x - a^n)*(a^p*z - a^3) = a^5*c^5) : 
  m * n * p = 18 := by
sorry

end equation_equivalence_implies_mnp_18_l1874_187470


namespace converse_of_quadratic_equation_l1874_187459

theorem converse_of_quadratic_equation (x : ℝ) : x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0 := by
  sorry

end converse_of_quadratic_equation_l1874_187459


namespace sixteen_power_division_plus_two_l1874_187473

theorem sixteen_power_division_plus_two (m : ℕ) : 
  m = 16^2023 → m / 8 + 2 = 2^8089 + 2 := by
  sorry

end sixteen_power_division_plus_two_l1874_187473


namespace vegetable_processing_plant_profit_l1874_187484

/-- Represents the total net profit for the first n years -/
def f (n : ℕ) : ℚ :=
  500000 * n - (120000 * n + 40000 * n * (n - 1) / 2) - 720000

/-- Represents the annual average net profit for the first n years -/
def avg_profit (n : ℕ) : ℚ := f n / n

theorem vegetable_processing_plant_profit :
  (∀ k : ℕ, k < 3 → f k ≤ 0) ∧
  f 3 > 0 ∧
  (∀ n : ℕ, n > 0 → avg_profit n ≤ avg_profit 6) ∧
  f 6 = 1440000 := by sorry

end vegetable_processing_plant_profit_l1874_187484


namespace complex_power_modulus_l1874_187476

theorem complex_power_modulus : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end complex_power_modulus_l1874_187476


namespace circle_center_apollonius_l1874_187463

/-- The center of the circle formed by points P where OP:PQ = 5:4, given O(0,0) and Q(1,2) -/
theorem circle_center_apollonius (P : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (1, 2)
  let OP := Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)
  let PQ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (4 * OP = 5 * PQ) → (25/9, 50/9) = (
    (P.1^2 + P.2^2 - 25 * ((P.1 - 1)^2 + (P.2 - 2)^2) / 16) / (2 * P.1),
    (P.1^2 + P.2^2 - 25 * ((P.1 - 1)^2 + (P.2 - 2)^2) / 16) / (2 * P.2)
  ) := by
sorry


end circle_center_apollonius_l1874_187463


namespace yanna_shirt_purchase_l1874_187488

theorem yanna_shirt_purchase (shirt_price : ℕ) (sandals_cost : ℕ) (total_spent : ℕ) 
  (h1 : shirt_price = 5)
  (h2 : sandals_cost = 9)
  (h3 : total_spent = 59) :
  ∃ (num_shirts : ℕ), num_shirts * shirt_price + sandals_cost = total_spent ∧ num_shirts = 10 := by
  sorry

end yanna_shirt_purchase_l1874_187488


namespace total_marbles_l1874_187497

/-- Proves that the total number of marbles is 4.51b given the conditions -/
theorem total_marbles (b : ℝ) (h1 : b > 0) : 
  let r := 1.3 * b
  let g := 1.7 * r
  b + r + g = 4.51 * b := by
  sorry

end total_marbles_l1874_187497


namespace events_mutually_exclusive_l1874_187472

theorem events_mutually_exclusive (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 1/5)
  (hB : P B = 1/3)
  (hAB : P (A ∪ B) = 8/15) :
  P (A ∪ B) = P A + P B :=
by
  sorry

end events_mutually_exclusive_l1874_187472
