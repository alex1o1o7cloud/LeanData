import Mathlib

namespace sequence_sum_property_l3846_384676

theorem sequence_sum_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℕ+) :
  (∀ n : ℕ+, S n = a n / n) →
  (1 < S k ∧ S k < 9) →
  k = 4 := by
sorry

end sequence_sum_property_l3846_384676


namespace expression_simplification_l3846_384674

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (3 * x * y)⁻¹ := by
  sorry

end expression_simplification_l3846_384674


namespace partial_fraction_decomposition_l3846_384638

theorem partial_fraction_decomposition :
  ∃ (A B C D : ℚ),
    (A = 1/15) ∧ (B = 5/2) ∧ (C = -59/6) ∧ (D = 42/5) ∧
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 →
      (x^3 - 7) / ((x - 2) * (x - 3) * (x - 5) * (x - 7)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x - 7)) :=
by
  sorry

end partial_fraction_decomposition_l3846_384638


namespace one_root_cubic_equation_a_range_l3846_384613

theorem one_root_cubic_equation_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-3*a)*x^2 + 2*a^2*x - 2*a*x + x + a^2 - a = 0) → 
  (-Real.sqrt 3 / 2 < a ∧ a < Real.sqrt 3 / 2) := by
  sorry

end one_root_cubic_equation_a_range_l3846_384613


namespace set_equality_implies_values_l3846_384637

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + (a-1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 2 = 0}

theorem set_equality_implies_values (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end set_equality_implies_values_l3846_384637


namespace parking_lot_problem_l3846_384619

theorem parking_lot_problem :
  ∀ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = 36 →
    6 * medium_cars + 4 * small_cars = 176 →
    medium_cars = 16 ∧ small_cars = 20 := by
  sorry

end parking_lot_problem_l3846_384619


namespace geric_bills_l3846_384683

theorem geric_bills (jessa kyla geric : ℕ) : 
  geric = 2 * kyla →
  kyla = jessa - 2 →
  jessa - 3 = 7 →
  geric = 16 := by
sorry

end geric_bills_l3846_384683


namespace bake_sale_group_composition_l3846_384600

theorem bake_sale_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 = (total * 2) / 5) →  -- After changes, 40% are girls
  (initial_girls = 15) :=
by
  sorry

#check bake_sale_group_composition

end bake_sale_group_composition_l3846_384600


namespace tenth_term_of_sequence_l3846_384675

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem tenth_term_of_sequence
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2)
  (h3 : a 1 = 1) :
  a 10 = 19 := by
sorry

end tenth_term_of_sequence_l3846_384675


namespace decagon_triangle_probability_l3846_384634

-- Define a regular decagon
structure RegularDecagon :=
  (vertices : Finset (ℕ × ℕ))
  (is_regular : vertices.card = 10)

-- Define a triangle formed by three vertices of the decagon
def Triangle (d : RegularDecagon) :=
  {t : Finset (ℕ × ℕ) // t ⊆ d.vertices ∧ t.card = 3}

-- Define a predicate for a triangle not sharing sides with the decagon
def NoSharedSides (d : RegularDecagon) (t : Triangle d) : Prop := sorry

-- Define the probability function
def Probability (d : RegularDecagon) : ℚ := sorry

-- State the theorem
theorem decagon_triangle_probability (d : RegularDecagon) :
  Probability d = 5 / 12 := by sorry

end decagon_triangle_probability_l3846_384634


namespace kishore_savings_l3846_384628

/-- Proves that given the total expenses and the fact that they represent 90% of the salary,
    the 10% savings amount to the correct value. -/
theorem kishore_savings (total_expenses : ℕ) (monthly_salary : ℕ) : 
  total_expenses = 20700 →
  total_expenses = (90 * monthly_salary) / 100 →
  (10 * monthly_salary) / 100 = 2300 :=
by sorry

end kishore_savings_l3846_384628


namespace bill_difference_l3846_384615

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 2)
  (h2 : joe_tip = 2)
  (h3 : mike_percent = 0.1)
  (h4 : joe_percent = 0.2)
  (h5 : mike_tip = mike_percent * mike_bill)
  (h6 : joe_tip = joe_percent * joe_bill)
  : mike_bill - joe_bill = 10 := by
  sorry

end bill_difference_l3846_384615


namespace radio_price_reduction_l3846_384663

theorem radio_price_reduction (x : ℝ) :
  (∀ (P Q : ℝ), P > 0 ∧ Q > 0 →
    P * (1 - x / 100) * (Q * 1.8) = P * Q * 1.44) →
  x = 20 := by
sorry

end radio_price_reduction_l3846_384663


namespace number_of_men_is_group_size_l3846_384697

/-- Represents the number of men, women, and boys -/
def group_size : ℕ := 8

/-- Represents the total earnings of all people -/
def total_earnings : ℕ := 105

/-- Represents the wage of each man -/
def men_wage : ℕ := 7

/-- Theorem stating that the number of men is equal to the group size -/
theorem number_of_men_is_group_size :
  ∃ (women_wage boy_wage : ℚ),
    group_size * men_wage + group_size * women_wage + group_size * boy_wage = total_earnings →
    group_size = group_size := by
  sorry

end number_of_men_is_group_size_l3846_384697


namespace sequences_theorem_l3846_384699

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 2

/-- Sequence b satisfying b_{n+1} - b_n = a_n -/
def b_sequence (a b : ℕ → ℤ) : Prop :=
  ∀ n, b (n + 1) - b n = a n

/-- Main theorem about sequences a and b -/
theorem sequences_theorem (a b : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : b_sequence a b)
  (h3 : b 2 = -18)
  (h4 : b 3 = -24) :
  (∀ n, a n = 2 * n - 10) ∧
  (b 5 = -30 ∧ b 6 = -30 ∧ ∀ n, b n ≥ -30) :=
sorry

end sequences_theorem_l3846_384699


namespace third_class_proportion_l3846_384631

theorem third_class_proportion (first_class second_class third_class : ℕ) 
  (h1 : first_class = 30)
  (h2 : second_class = 50)
  (h3 : third_class = 20) :
  (third_class : ℚ) / (first_class + second_class + third_class : ℚ) = 0.2 := by
  sorry

end third_class_proportion_l3846_384631


namespace fixed_point_exists_l3846_384658

/-- For any a > 0 and a ≠ 1, the function f(x) = ax - 5 has a fixed point at x = 2 -/
theorem fixed_point_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a * x - 5 = x ∧ x = 2 := by
  sorry

end fixed_point_exists_l3846_384658


namespace geometric_sequence_sum_l3846_384616

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 2 * a 6 + 2 * a 4 * a 5 + a 1 * a 9 = 25 →
  a 4 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l3846_384616


namespace polynomial_division_degree_l3846_384678

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (f = d * q + r) →
  (Polynomial.degree q = 8) →
  (r = 5 * X^4 + 3 * X^2 - 2 * X + 7) →
  (Polynomial.degree d = 7) := by
sorry

end polynomial_division_degree_l3846_384678


namespace six_balls_three_boxes_l3846_384655

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem: There are 10 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball. -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 10 := by
  sorry

#eval distribute_balls 6 3

end six_balls_three_boxes_l3846_384655


namespace initially_calculated_average_weight_l3846_384601

/-- Given a class of boys with a misread weight, prove the initially calculated average weight. -/
theorem initially_calculated_average_weight
  (n : ℕ) -- number of boys
  (correct_avg : ℝ) -- correct average weight
  (misread_weight : ℝ) -- misread weight
  (correct_weight : ℝ) -- correct weight
  (h1 : n = 20)
  (h2 : correct_avg = 58.7)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 := by
  sorry

end initially_calculated_average_weight_l3846_384601


namespace tangent_line_to_sqrt_curve_l3846_384672

theorem tangent_line_to_sqrt_curve (x y : ℝ) :
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    (a * 1 + b * 2 + c = 0) ∧
    (∃ (x₀ : ℝ), x₀ > 0 ∧ 
      a * x₀ + b * Real.sqrt x₀ + c = 0 ∧
      a + b * (1 / (2 * Real.sqrt x₀)) = 0)) ↔
  ((x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0) ∨
   (x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)) :=
by sorry

end tangent_line_to_sqrt_curve_l3846_384672


namespace roots_of_polynomial_l3846_384681

-- Define the polynomial
def p (x : ℝ) : ℝ := (x^2 - 5*x + 6) * x * (x - 4) * (x - 6)

-- State the theorem
theorem roots_of_polynomial : 
  {x : ℝ | p x = 0} = {0, 2, 3, 4, 6} := by sorry

end roots_of_polynomial_l3846_384681


namespace nine_points_interior_lattice_point_l3846_384650

/-- A lattice point in 3D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The statement that there exists an interior lattice point -/
def exists_interior_lattice_point (points : Finset LatticePoint) : Prop :=
  ∃ p q : LatticePoint, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    ∃ r : LatticePoint, r.x = (p.x + q.x) / 2 ∧ 
                        r.y = (p.y + q.y) / 2 ∧ 
                        r.z = (p.z + q.z) / 2

/-- The main theorem -/
theorem nine_points_interior_lattice_point 
  (points : Finset LatticePoint) 
  (h : points.card = 9) : 
  exists_interior_lattice_point points := by
  sorry

#check nine_points_interior_lattice_point

end nine_points_interior_lattice_point_l3846_384650


namespace and_sufficient_not_necessary_for_or_l3846_384606

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (r s : Prop), (r ∨ s) ∧ ¬(r ∧ s) := by
  sorry

end and_sufficient_not_necessary_for_or_l3846_384606


namespace skirt_cut_amount_l3846_384679

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end skirt_cut_amount_l3846_384679


namespace simplify_and_rationalize_l3846_384629

theorem simplify_and_rationalize : 
  (Real.sqrt 8 / Real.sqrt 3) * (Real.sqrt 25 / Real.sqrt 30) * (Real.sqrt 16 / Real.sqrt 21) = 
  4 * Real.sqrt 14 / 63 := by sorry

end simplify_and_rationalize_l3846_384629


namespace cryptarithm_solution_exists_l3846_384673

theorem cryptarithm_solution_exists : ∃ (Φ E B P A J : ℕ), 
  Φ < 10 ∧ E < 10 ∧ B < 10 ∧ P < 10 ∧ A < 10 ∧ J < 10 ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J ∧
  E ≠ 0 ∧ A ≠ 0 ∧ J ≠ 0 ∧
  (Φ : ℚ) / E + (B * 10 + P : ℚ) / (A * J) = 1 := by
  sorry

end cryptarithm_solution_exists_l3846_384673


namespace number_equation_solution_l3846_384698

theorem number_equation_solution : ∃ x : ℝ, (7 * x = 3 * x + 12) ∧ (x = 3) := by
  sorry

end number_equation_solution_l3846_384698


namespace third_turtle_lying_l3846_384611

-- Define the type for turtles
inductive Turtle : Type
  | T1 : Turtle
  | T2 : Turtle
  | T3 : Turtle

-- Define the relative position of turtles
inductive Position : Type
  | Front : Position
  | Behind : Position

-- Define a function to represent the statement of each turtle
def turtleStatement (t : Turtle) : List (Turtle × Position) :=
  match t with
  | Turtle.T1 => [(Turtle.T2, Position.Behind), (Turtle.T3, Position.Behind)]
  | Turtle.T2 => [(Turtle.T1, Position.Front), (Turtle.T3, Position.Behind)]
  | Turtle.T3 => [(Turtle.T1, Position.Front), (Turtle.T2, Position.Front), (Turtle.T3, Position.Behind)]

-- Define a function to check if a turtle's statement is consistent with its position
def isConsistent (t : Turtle) (position : Nat) : Prop :=
  match t, position with
  | Turtle.T1, 0 => true
  | Turtle.T2, 1 => true
  | Turtle.T3, 2 => false
  | _, _ => false

-- Theorem: The third turtle's statement is inconsistent
theorem third_turtle_lying :
  ¬ (isConsistent Turtle.T3 2) :=
  sorry


end third_turtle_lying_l3846_384611


namespace rectangle_width_length_ratio_l3846_384691

theorem rectangle_width_length_ratio :
  ∀ w : ℝ,
  w > 0 →
  2 * w + 2 * 10 = 30 →
  w / 10 = 1 / 2 := by
sorry

end rectangle_width_length_ratio_l3846_384691


namespace factor_divisor_statements_l3846_384618

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem factor_divisor_statements : 
  (is_factor 5 25) ∧ 
  (is_divisor 19 209 ∧ ¬is_divisor 19 63) ∧ 
  (is_divisor 20 80) ∧ 
  (is_divisor 14 28 ∧ is_divisor 14 56) ∧ 
  (is_factor 7 140) := by
  sorry

end factor_divisor_statements_l3846_384618


namespace chicken_cost_per_person_l3846_384649

def grocery_cost : ℝ := 16
def beef_price_per_pound : ℝ := 4
def beef_pounds : ℝ := 3
def oil_price : ℝ := 1
def number_of_people : ℕ := 3

theorem chicken_cost_per_person (chicken_cost : ℝ) : 
  chicken_cost = grocery_cost - (beef_price_per_pound * beef_pounds + oil_price) →
  chicken_cost / number_of_people = 1 := by sorry

end chicken_cost_per_person_l3846_384649


namespace weavers_count_proof_l3846_384657

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 6

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 9

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 6

theorem weavers_count_proof :
  (first_group_mats : ℚ) / (first_group_weavers * first_group_days) =
  (second_group_mats : ℚ) / (second_group_weavers * second_group_days) →
  first_group_weavers = 4 := by
  sorry

end weavers_count_proof_l3846_384657


namespace area_between_circles_l3846_384622

/-- The area between two concentric circles, where the larger circle's radius is three times 
    the smaller circle's radius, and the smaller circle's diameter is 6 units, 
    is equal to 72π square units. -/
theorem area_between_circles (π : ℝ) : 
  let small_diameter : ℝ := 6
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := 3 * small_radius
  let area_large : ℝ := π * large_radius ^ 2
  let area_small : ℝ := π * small_radius ^ 2
  area_large - area_small = 72 * π := by
sorry

end area_between_circles_l3846_384622


namespace factorization_of_x2y_minus_4y_l3846_384633

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end factorization_of_x2y_minus_4y_l3846_384633


namespace reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l3846_384610

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Statement 1
theorem reciprocals_multiply_to_one (x y : ℝ) :
  are_reciprocals x y → x * y = 1 :=
sorry

-- Definition for real roots
def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

-- Statement 2
theorem no_real_roots_when_m_greater_than_one (m : ℝ) :
  m > 1 → ¬(has_real_roots 1 (-2) m) :=
sorry

end reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l3846_384610


namespace promotion_equivalence_bottles_in_box_l3846_384693

/-- The cost of a box of beverage in yuan -/
def box_cost : ℝ := 26

/-- The discount per bottle in yuan due to the promotion -/
def discount_per_bottle : ℝ := 0.6

/-- The number of free bottles given in the promotion -/
def free_bottles : ℕ := 3

/-- The number of bottles in a box -/
def bottles_per_box : ℕ := 10

theorem promotion_equivalence : 
  (box_cost / bottles_per_box) - (box_cost / (bottles_per_box + free_bottles)) = discount_per_bottle :=
sorry

theorem bottles_in_box : 
  bottles_per_box = 10 :=
sorry

end promotion_equivalence_bottles_in_box_l3846_384693


namespace sequence_is_geometric_from_second_term_l3846_384603

def is_geometric_from_second_term (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n + 1) = r * a n

theorem sequence_is_geometric_from_second_term
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S 1 = 1)
  (h2 : S 2 = 2)
  (h3 : ∀ n : ℕ, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0)
  (h4 : ∀ n : ℕ, S (n + 1) - S n = a (n + 1))
  : is_geometric_from_second_term a :=
by
  sorry

#check sequence_is_geometric_from_second_term

end sequence_is_geometric_from_second_term_l3846_384603


namespace gcd_of_72_120_168_l3846_384652

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l3846_384652


namespace exists_k_undecided_tournament_l3846_384632

/-- A tournament is represented as a function that takes two players and returns true if the first player defeats the second, and false otherwise. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for any set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → t p a = true

/-- For any positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ), n > k ∧ ∃ (t : Tournament n), IsKUndecided k n t :=
sorry

end exists_k_undecided_tournament_l3846_384632


namespace median_trigonometric_values_max_condition_implies_range_l3846_384647

def median (a b c : ℝ) : ℝ := sorry

def max3 (a b c : ℝ) : ℝ := sorry

theorem median_trigonometric_values :
  median (Real.sin (30 * π / 180)) (Real.cos (45 * π / 180)) (Real.tan (60 * π / 180)) = Real.sqrt 2 / 2 := by sorry

theorem max_condition_implies_range (x : ℝ) :
  max3 5 (2*x - 3) (-10 - 3*x) = 5 → -5 ≤ x ∧ x ≤ 4 := by sorry

end median_trigonometric_values_max_condition_implies_range_l3846_384647


namespace x_equals_one_sufficient_not_necessary_l3846_384659

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) :=
by sorry

end x_equals_one_sufficient_not_necessary_l3846_384659


namespace car_trade_profit_percentage_l3846_384680

/-- Calculates the profit percentage on the original price when a car is bought at a discount and sold at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : discount_percentage = 20) 
  (h2 : increase_percentage = 50) 
  : (((1 - discount_percentage / 100) * (1 + increase_percentage / 100) - 1) * 100 = 20) := by
sorry

end car_trade_profit_percentage_l3846_384680


namespace sum_of_c_values_l3846_384627

theorem sum_of_c_values (b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ (z = x ∨ z = y)) →
  b = c - 1 →
  ∃ c₁ c₂ : ℝ, (∀ c' : ℝ, (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + (c' - 1)*z + c' = 0 ↔ (z = x ∨ z = y)) ↔ (c' = c₁ ∨ c' = c₂)) ∧
  c₁ + c₂ = 6 :=
by sorry

end sum_of_c_values_l3846_384627


namespace triangle_side_count_l3846_384654

theorem triangle_side_count (a b : ℕ) (ha : a = 8) (hb : b = 5) :
  ∃! n : ℕ, n = (Finset.range (a + b - 1) \ Finset.range (a - b + 1)).card :=
by sorry

end triangle_side_count_l3846_384654


namespace exists_large_ratio_l3846_384620

def sequence_property (a b : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0 ∧ b n > 0) ∧
  (∀ n : ℕ+, a (n + 1) * b (n + 1) = a n ^ 2 + b n ^ 2) ∧
  (∀ n : ℕ+, a (n + 1) + b (n + 1) = a n * b n) ∧
  (∀ n : ℕ+, a n ≥ b n)

theorem exists_large_ratio (a b : ℕ+ → ℝ) (h : sequence_property a b) :
  ∃ n : ℕ+, a n / b n > 2023^2023 := by
  sorry

end exists_large_ratio_l3846_384620


namespace remainder_problem_l3846_384639

theorem remainder_problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 1816 = k * x + 6) : 
  ∃ l : ℕ, 1442 = l * x + 0 := by
  sorry

end remainder_problem_l3846_384639


namespace marble_distribution_l3846_384690

theorem marble_distribution (x : ℕ) 
  (liam : ℕ) (mia : ℕ) (noah : ℕ) (olivia : ℕ) : 
  liam = x ∧ 
  mia = 3 * x ∧ 
  noah = 12 * x ∧ 
  olivia = 24 * x ∧ 
  liam + mia + noah + olivia = 160 → 
  x = 4 := by
sorry

end marble_distribution_l3846_384690


namespace sin_sixteen_thirds_pi_l3846_384648

theorem sin_sixteen_thirds_pi : Real.sin (16 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_sixteen_thirds_pi_l3846_384648


namespace expression_value_l3846_384661

theorem expression_value (x y : ℝ) (h1 : x ≠ y) (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end expression_value_l3846_384661


namespace factors_multiple_of_180_l3846_384607

/-- The number of natural-number factors of m that are multiples of 180 -/
def count_factors (m : ℕ) : ℕ :=
  sorry

theorem factors_multiple_of_180 :
  let m : ℕ := 2^12 * 3^15 * 5^9
  count_factors m = 1386 := by
  sorry

end factors_multiple_of_180_l3846_384607


namespace box_neg_two_two_neg_one_l3846_384644

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

theorem box_neg_two_two_neg_one : box (-2) 2 (-1) = (7 / 2 : ℚ) := by
  sorry

end box_neg_two_two_neg_one_l3846_384644


namespace amys_candy_problem_l3846_384608

/-- Amy's candy problem -/
theorem amys_candy_problem (candy_given : ℕ) (difference : ℕ) : 
  candy_given = 6 → difference = 1 → candy_given - difference = 5 := by
  sorry

end amys_candy_problem_l3846_384608


namespace minimum_at_two_implies_m_geq_five_range_of_m_l3846_384605

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - 1| + m * |x - 2| + 6 * |x - 3|

/-- The theorem stating that if f attains its minimum at x = 2, then m ≥ 5 -/
theorem minimum_at_two_implies_m_geq_five (m : ℝ) :
  (∀ x : ℝ, f m x ≥ f m 2) → m ≥ 5 := by
  sorry

/-- The main theorem describing the range of m -/
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, f m x ≥ f m 2} = {m : ℝ | m ≥ 5} := by
  sorry

end minimum_at_two_implies_m_geq_five_range_of_m_l3846_384605


namespace systematic_sampling_result_l3846_384614

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startingNumber : ℕ

/-- Generates the list of selected student numbers -/
def generateSample (s : SystematicSampling) : List ℕ :=
  let interval := s.totalStudents / s.sampleSize
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * interval)

theorem systematic_sampling_result :
  ∀ (s : SystematicSampling),
    s.totalStudents = 50 →
    s.sampleSize = 5 →
    s.startingNumber = 3 →
    generateSample s = [3, 13, 23, 33, 43] :=
by
  sorry

#eval generateSample ⟨50, 5, 3⟩

end systematic_sampling_result_l3846_384614


namespace straws_paper_difference_l3846_384692

theorem straws_paper_difference :
  let straws : ℕ := 15
  let paper : ℕ := 7
  straws - paper = 8 := by sorry

end straws_paper_difference_l3846_384692


namespace set_union_problem_l3846_384668

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
  sorry

end set_union_problem_l3846_384668


namespace subcommittee_formation_ways_senate_subcommittee_formation_l3846_384640

theorem subcommittee_formation_ways (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) : Nat :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem senate_subcommittee_formation : 
  subcommittee_formation_ways 10 8 4 3 = 11760 := by
  sorry

end subcommittee_formation_ways_senate_subcommittee_formation_l3846_384640


namespace cosine_sum_pentagon_l3846_384682

theorem cosine_sum_pentagon : 
  Real.cos (5 * π / 180) + Real.cos (77 * π / 180) + Real.cos (149 * π / 180) + 
  Real.cos (221 * π / 180) + Real.cos (293 * π / 180) = 0 := by
  sorry

end cosine_sum_pentagon_l3846_384682


namespace sum_of_sixth_root_arguments_l3846_384635

open Complex

/-- The complex number whose sixth power is equal to -1/√3 - i√(2/3) -/
def z : ℂ := sorry

/-- The argument of z^6 in radians -/
def arg_z6 : ℝ := sorry

/-- The list of arguments of the sixth roots of z^6 in radians -/
def root_args : List ℝ := sorry

theorem sum_of_sixth_root_arguments :
  (root_args.sum * (180 / Real.pi)) = 1140 ∧ 
  (∀ φ ∈ root_args, 0 ≤ φ * (180 / Real.pi) ∧ φ * (180 / Real.pi) < 360) ∧
  (List.length root_args = 6) ∧
  (∀ φ ∈ root_args, Complex.exp (φ * Complex.I) ^ 6 = z^6) := by
  sorry

end sum_of_sixth_root_arguments_l3846_384635


namespace smallest_angle_in_ratio_triangle_l3846_384671

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 →
  (a : ℝ) / 4 = (c : ℝ) / 7 →
  a + b + c = 180 →
  a = 45 := by
sorry

end smallest_angle_in_ratio_triangle_l3846_384671


namespace pigeonhole_socks_l3846_384685

theorem pigeonhole_socks (red blue : ℕ) (h1 : red = 10) (h2 : blue = 10) :
  ∃ n : ℕ, n = 3 ∧ 
  (∀ m : ℕ, m < n → ∃ f : Fin m → Bool, Function.Injective f) ∧
  (∀ f : Fin n → Bool, ¬Function.Injective f) :=
sorry

end pigeonhole_socks_l3846_384685


namespace waiter_customers_l3846_384602

theorem waiter_customers (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping_customers = 5 →
  tip_amount = 8 →
  total_tips = 32 →
  non_tipping_customers + (total_tips / tip_amount) = 9 :=
by sorry

end waiter_customers_l3846_384602


namespace incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l3846_384667

/-- Represents the color of a bear -/
inductive BearColor
| White
| Brown
| Black

/-- Represents a row of 1000 bears -/
def BearRow := Fin 1000 → BearColor

/-- Predicate to check if three consecutive bears have all three colors -/
def hasAllColors (row : BearRow) (i : Fin 998) : Prop :=
  ∃ (c1 c2 c3 : BearColor), 
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    row i = c1 ∧ row (i + 1) = c2 ∧ row (i + 2) = c3

/-- The main theorem stating that the 20th bear's color must be the incorrect guess -/
theorem incorrect_guess_is_20th_bear (row : BearRow) : Prop :=
  (∀ i : Fin 998, hasAllColors row i) →
  (row 1 = BearColor.White) →
  (row 399 = BearColor.Black) →
  (row 599 = BearColor.Brown) →
  (row 799 = BearColor.White) →
  (row 19 ≠ BearColor.Brown)

-- The proof of the theorem
theorem prove_incorrect_guess_is_20th_bear :
  ∃ (row : BearRow), incorrect_guess_is_20th_bear row :=
sorry

end incorrect_guess_is_20th_bear_prove_incorrect_guess_is_20th_bear_l3846_384667


namespace angle_function_value_l3846_384625

/-- Given m < 0 and point M(3m, -m) on the terminal side of angle α, 
    prove that 1 / (2sin(α)cos(α) + cos²(α)) = 10/3 -/
theorem angle_function_value (m : ℝ) (α : ℝ) :
  m < 0 →
  let M : ℝ × ℝ := (3 * m, -m)
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 10 / 3 := by
  sorry

end angle_function_value_l3846_384625


namespace circle1_properties_circle2_properties_l3846_384621

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0
def circle2 (x y : ℝ) : Prop := 3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -2 ∧ r = 3) ∧
    ∀ (x y : ℝ), circle1 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -1/2 ∧ r = 5/2) ∧
    ∀ (x y : ℝ), circle2 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end circle1_properties_circle2_properties_l3846_384621


namespace edward_earnings_l3846_384642

theorem edward_earnings : 
  let lawn_pay : ℕ := 8
  let garden_pay : ℕ := 12
  let lawns_mowed : ℕ := 5
  let gardens_cleaned : ℕ := 3
  let fuel_cost : ℕ := 10
  let equipment_cost : ℕ := 15
  let initial_savings : ℕ := 7

  let total_earnings := lawn_pay * lawns_mowed + garden_pay * gardens_cleaned
  let total_expenses := fuel_cost + equipment_cost
  let final_amount := total_earnings + initial_savings - total_expenses

  final_amount = 58 := by sorry

end edward_earnings_l3846_384642


namespace sum_of_unique_decimals_sum_of_unique_decimals_proof_l3846_384630

/-- The sum of all unique decimals formed by 4 distinct digit cards and 1 decimal point card -/
theorem sum_of_unique_decimals : ℝ :=
  let digit_sum := (0 : ℕ) + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
  let num_permutations := 24
  let num_decimal_positions := 4
  666.6

/-- The number of unique decimals that can be formed -/
def num_unique_decimals : ℕ := 72

theorem sum_of_unique_decimals_proof :
  sum_of_unique_decimals = 666.6 ∧ num_unique_decimals = 72 := by
  sorry

end sum_of_unique_decimals_sum_of_unique_decimals_proof_l3846_384630


namespace max_sum_of_factors_l3846_384653

theorem max_sum_of_factors (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 144 →
  a + b + c ≤ 75 :=
by sorry

end max_sum_of_factors_l3846_384653


namespace janes_bowling_score_l3846_384694

def janes_score (x : ℝ) := x
def toms_score (x : ℝ) := x - 50

theorem janes_bowling_score :
  ∀ x : ℝ,
  janes_score x = toms_score x + 50 →
  (janes_score x + toms_score x) / 2 = 90 →
  janes_score x = 115 :=
by
  sorry

end janes_bowling_score_l3846_384694


namespace smartpup_academy_total_dogs_l3846_384665

/-- Represents the number of dogs at Smartpup Tricks Academy with various skill combinations -/
structure DogSkills where
  fetch : ℕ
  jump : ℕ
  play_dead : ℕ
  fetch_and_jump : ℕ
  jump_and_play_dead : ℕ
  fetch_and_play_dead : ℕ
  all_three : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the academy -/
def total_dogs (skills : DogSkills) : ℕ :=
  skills.all_three +
  (skills.fetch_and_play_dead - skills.all_three) +
  (skills.jump_and_play_dead - skills.all_three) +
  (skills.fetch_and_jump - skills.all_three) +
  (skills.fetch - skills.fetch_and_jump - skills.fetch_and_play_dead + skills.all_three) +
  (skills.jump - skills.fetch_and_jump - skills.jump_and_play_dead + skills.all_three) +
  (skills.play_dead - skills.fetch_and_play_dead - skills.jump_and_play_dead + skills.all_three) +
  skills.none

/-- The main theorem stating that the total number of dogs is 75 -/
theorem smartpup_academy_total_dogs :
  let skills : DogSkills := {
    fetch := 40,
    jump := 35,
    play_dead := 22,
    fetch_and_jump := 14,
    jump_and_play_dead := 10,
    fetch_and_play_dead := 16,
    all_three := 6,
    none := 12
  }
  total_dogs skills = 75 := by
  sorry

end smartpup_academy_total_dogs_l3846_384665


namespace factor_polynomial_l3846_384669

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 180 * x^9 = 20 * x^5 * (4 - 9 * x^4) := by
  sorry

end factor_polynomial_l3846_384669


namespace longest_line_segment_in_quarter_pie_l3846_384604

theorem longest_line_segment_in_quarter_pie (d : ℝ) (h : d = 16) :
  let r := d / 2
  let θ := π / 2
  let chord_length := 2 * r * Real.sin (θ / 2)
  chord_length ^ 2 = 128 :=
by sorry

end longest_line_segment_in_quarter_pie_l3846_384604


namespace arithmetic_mean_after_removal_l3846_384617

theorem arithmetic_mean_after_removal (S : Finset ℝ) (a b c : ℝ) :
  S.card = 60 →
  a = 48 ∧ b = 52 ∧ c = 56 →
  a ∈ S ∧ b ∈ S ∧ c ∈ S →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (a + b + c)) / (S.card - 3) = 41.47 := by
sorry

end arithmetic_mean_after_removal_l3846_384617


namespace complex_magnitude_theorem_l3846_384623

theorem complex_magnitude_theorem (b : ℝ) :
  (Complex.I * Complex.I.re = ((1 + b * Complex.I) * (2 + Complex.I)).re) →
  Complex.abs ((2 * b + 3 * Complex.I) / (1 + b * Complex.I)) = Real.sqrt 5 := by
  sorry

end complex_magnitude_theorem_l3846_384623


namespace unique_factors_of_2013_l3846_384670

theorem unique_factors_of_2013 (m n : ℕ) (h1 : m < n) (h2 : n < 2 * m) (h3 : m * n = 2013) :
  m = 33 ∧ n = 61 :=
sorry

end unique_factors_of_2013_l3846_384670


namespace johnny_marble_selection_l3846_384645

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of marbles in the collection -/
def total_marbles : ℕ := 10

/-- The number of marbles chosen in the first step -/
def first_choice : ℕ := 4

/-- The number of marbles chosen in the second step -/
def second_choice : ℕ := 2

/-- The theorem stating the total number of ways Johnny can complete the selection process -/
theorem johnny_marble_selection :
  (choose total_marbles first_choice) * (choose first_choice second_choice) = 1260 := by
  sorry

end johnny_marble_selection_l3846_384645


namespace expression_simplification_l3846_384687

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / ((1/2 * x * y)^2) = 8*x - 12*y := by
  sorry

end expression_simplification_l3846_384687


namespace work_ratio_l3846_384641

/-- The time (in days) it takes for worker A to complete the task alone -/
def time_A : ℝ := 6

/-- The time (in days) it takes for worker B to complete the task alone -/
def time_B : ℝ := 30

/-- The time (in days) it takes for workers A and B to complete the task together -/
def time_together : ℝ := 5

theorem work_ratio : 
  (1 / time_A + 1 / time_B = 1 / time_together) → 
  (time_A / time_B = 1 / 5) := by
  sorry

end work_ratio_l3846_384641


namespace equation_solutions_l3846_384664

theorem equation_solutions :
  -- Equation 1
  (∀ x : ℝ, x^2 - 5*x = 0 ↔ x = 0 ∨ x = 5) ∧
  -- Equation 2
  (∀ x : ℝ, (2*x + 1)^2 = 4 ↔ x = -3/2 ∨ x = 1/2) ∧
  -- Equation 3
  (∀ x : ℝ, x*(x - 1) + 3*(x - 1) = 0 ↔ x = 1 ∨ x = -3) ∧
  -- Equation 4
  (∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) := by
  sorry


end equation_solutions_l3846_384664


namespace symmetric_complex_division_l3846_384686

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given two complex numbers z₁ and z₂ symmetric with respect to the imaginary axis,
    where z₁ = -1 + i, prove that z₁ / z₂ = i. -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
    (h_sym : symmetric_to_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I := by
  sorry


end symmetric_complex_division_l3846_384686


namespace log_27_3_l3846_384612

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end log_27_3_l3846_384612


namespace least_N_for_P_condition_l3846_384626

/-- The probability that at least 3/5 of N green balls are on the same side of a red ball
    in a random arrangement of N green balls and one red ball. -/
def P (N : ℕ) : ℚ :=
  (↑⌈(3 * N : ℚ) / 5 + 1⌉) / (N + 1)

/-- 480 is the least positive multiple of 5 for which P(N) < 321/400 -/
theorem least_N_for_P_condition : ∀ N : ℕ,
  N > 0 ∧ N % 5 = 0 ∧ P N < 321 / 400 → N ≥ 480 :=
sorry

end least_N_for_P_condition_l3846_384626


namespace bill_percentage_increase_l3846_384689

/-- 
Given Maximoff's original monthly bill and new monthly bill, 
prove that the percentage increase is 30%.
-/
theorem bill_percentage_increase 
  (original_bill : ℝ) 
  (new_bill : ℝ) 
  (h1 : original_bill = 60) 
  (h2 : new_bill = 78) : 
  (new_bill - original_bill) / original_bill * 100 = 30 := by
sorry

end bill_percentage_increase_l3846_384689


namespace line_through_origin_and_third_quadrant_l3846_384609

/-- A line in 2D space represented by the equation Ax - By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a point (x, y) lies on a given line -/
def Line.contains (L : Line) (x y : ℝ) : Prop :=
  L.A * x - L.B * y + L.C = 0

/-- Predicate to check if a line passes through the origin -/
def Line.passes_through_origin (L : Line) : Prop :=
  L.contains 0 0

/-- Predicate to check if a line passes through the third quadrant -/
def Line.passes_through_third_quadrant (L : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ L.contains x y

/-- Theorem stating the properties of a line passing through the origin and third quadrant -/
theorem line_through_origin_and_third_quadrant (L : Line) :
  L.passes_through_origin ∧ L.passes_through_third_quadrant →
  L.A * L.B < 0 ∧ L.C = 0 :=
by sorry

end line_through_origin_and_third_quadrant_l3846_384609


namespace equal_kite_areas_condition_l3846_384660

/-- An isosceles triangle with perpendiculars from the intersection point of angle bisectors -/
structure IsoscelesTriangleWithPerpendiculars where
  /-- Length of the legs of the isosceles triangle -/
  leg_length : ℝ
  /-- Length of the base of the isosceles triangle -/
  base_length : ℝ
  /-- The triangle is isosceles -/
  isosceles : leg_length > 0
  /-- The perpendiculars divide the triangle into two smaller kites and one larger kite -/
  has_kites : True

/-- The theorem stating the condition for equal areas of kites -/
theorem equal_kite_areas_condition (t : IsoscelesTriangleWithPerpendiculars) :
  (∃ (small_kite_area larger_kite_area : ℝ),
    small_kite_area > 0 ∧ larger_kite_area > 0 ∧
    2 * small_kite_area = larger_kite_area) ↔
  t.base_length = 2/3 * t.leg_length :=
sorry

end equal_kite_areas_condition_l3846_384660


namespace lake_superior_depth_l3846_384651

/-- The depth of a lake given its water surface elevation above sea level and lowest point below sea level -/
def lake_depth (water_surface_elevation : ℝ) (lowest_point_below_sea : ℝ) : ℝ :=
  water_surface_elevation + lowest_point_below_sea

/-- Theorem: The depth of Lake Superior at its deepest point is 400 meters -/
theorem lake_superior_depth :
  lake_depth 180 220 = 400 := by
  sorry

end lake_superior_depth_l3846_384651


namespace distance_between_3rd_and_21st_red_lights_l3846_384643

/-- Represents the pattern of lights on the string -/
inductive LightColor
| Red
| Green

/-- Defines the repeating pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 6

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Function to get the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the distance between the 3rd and 21st red lights -/
theorem distance_between_3rd_and_21st_red_lights :
  (nthRedLightPosition 21 - nthRedLightPosition 3) * lightSpacing / inchesPerFoot = 22 :=
sorry

end distance_between_3rd_and_21st_red_lights_l3846_384643


namespace min_value_expression_l3846_384666

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  3 * a^2 + 3 * b^2 + 1 / (a + b)^2 + 4 / (a^2 * b^2) ≥ 6 := by
  sorry

end min_value_expression_l3846_384666


namespace paul_reading_books_l3846_384695

/-- The number of books Paul reads per week -/
def books_per_week : ℕ := 7

/-- The number of weeks Paul reads -/
def weeks : ℕ := 12

/-- The total number of books Paul reads -/
def total_books : ℕ := books_per_week * weeks

theorem paul_reading_books : total_books = 84 := by
  sorry

end paul_reading_books_l3846_384695


namespace frequency_count_calculation_l3846_384684

/-- Given a sample of size 1000 divided into several groups,
    if the frequency of a particular group is 0.4,
    then the frequency count of that group is 400. -/
theorem frequency_count_calculation (sample_size : ℕ) (group_frequency : ℝ) :
  sample_size = 1000 →
  group_frequency = 0.4 →
  (sample_size : ℝ) * group_frequency = 400 := by
  sorry

end frequency_count_calculation_l3846_384684


namespace closest_integer_to_cube_root_l3846_384696

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by
  sorry

end closest_integer_to_cube_root_l3846_384696


namespace inequality_proof_l3846_384646

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ 
  Real.sqrt 2 / (a * b * c) := by
sorry

end inequality_proof_l3846_384646


namespace remaining_amount_l3846_384656

def initial_amount : ℝ := 100.00
def spent_amount : ℝ := 15.00

theorem remaining_amount :
  initial_amount - spent_amount = 85.00 := by
  sorry

end remaining_amount_l3846_384656


namespace quadratic_equation_roots_l3846_384688

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 := by
  sorry

end quadratic_equation_roots_l3846_384688


namespace admin_teacher_ratio_l3846_384624

/-- The ratio of administrators to teachers at a graduation ceremony -/
theorem admin_teacher_ratio :
  let graduates : ℕ := 50
  let parents_per_graduate : ℕ := 2
  let teachers : ℕ := 20
  let total_chairs : ℕ := 180
  let grad_parent_chairs := graduates * (parents_per_graduate + 1)
  let admin_chairs := total_chairs - (grad_parent_chairs + teachers)
  (admin_chairs : ℚ) / teachers = 1 / 2 := by
  sorry

end admin_teacher_ratio_l3846_384624


namespace travis_apples_count_l3846_384662

/-- The number of apples that fit in each box -/
def apples_per_box : ℕ := 50

/-- The price of each box of apples in dollars -/
def price_per_box : ℕ := 35

/-- The total amount Travis takes home in dollars -/
def total_revenue : ℕ := 7000

/-- The number of apples Travis has -/
def travis_apples : ℕ := total_revenue / price_per_box * apples_per_box

theorem travis_apples_count : travis_apples = 10000 := by
  sorry

end travis_apples_count_l3846_384662


namespace trajectory_is_straight_line_l3846_384677

/-- The trajectory of a point P(x, y) equidistant from M(-2, 0) and the line x = -2 is a straight line y = 0 -/
theorem trajectory_is_straight_line :
  ∀ (x y : ℝ), 
    (|x + 2| = Real.sqrt ((x + 2)^2 + y^2)) → 
    y = 0 := by
  sorry

end trajectory_is_straight_line_l3846_384677


namespace billboard_count_l3846_384636

theorem billboard_count (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (total_hours : ℕ) (avg : ℕ) 
  (h1_count : h1 = 17)
  (h2_count : h2 = 20)
  (hours : total_hours = 3)
  (average : avg = 20)
  (avg_def : avg * total_hours = h1 + h2 + h3) :
  h3 = 23 := by
sorry

end billboard_count_l3846_384636
