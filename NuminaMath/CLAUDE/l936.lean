import Mathlib

namespace f_min_value_l936_93629

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

-- State the theorem
theorem f_min_value :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 3) :=
sorry

end f_min_value_l936_93629


namespace negation_of_proposition_negation_of_specific_proposition_l936_93678

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l936_93678


namespace postage_calculation_l936_93637

/-- Calculates the postage cost for a letter based on its weight and given rates. -/
def calculatePostage (weight : ℚ) (baseRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  baseRate + additionalCharges * additionalRate

/-- Theorem stating that the postage for a 4.5-ounce letter is 1.18 dollars 
    given the specified rates. -/
theorem postage_calculation :
  let weight : ℚ := 4.5
  let baseRate : ℚ := 0.30
  let additionalRate : ℚ := 0.22
  calculatePostage weight baseRate additionalRate = 1.18 := by
  sorry


end postage_calculation_l936_93637


namespace smallest_factorial_divisor_l936_93650

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_factorial_divisor (n : ℕ) (h1 : n > 1) :
  (∀ k : ℕ, k > 1 ∧ k < 7 → ¬(factorial k % n = 0)) ∧ (factorial 7 % n = 0) →
  n = 7 := by
  sorry

end smallest_factorial_divisor_l936_93650


namespace quadratic_roots_average_l936_93669

theorem quadratic_roots_average (d : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0) :
  (∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0 ∧ (x + y) / 2 = 1.5) :=
by
  sorry

end quadratic_roots_average_l936_93669


namespace sqrt_450_simplification_l936_93688

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l936_93688


namespace no_primes_satisfy_equation_l936_93653

theorem no_primes_satisfy_equation :
  ∀ (p q : ℕ) (n : ℕ+), 
    Prime p → Prime q → p ≠ q → p^(q-1) - q^(p-1) ≠ 4*(n:ℕ)^2 := by
  sorry

end no_primes_satisfy_equation_l936_93653


namespace divisibility_problem_l936_93603

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a.val := by
sorry

end divisibility_problem_l936_93603


namespace quadratic_inequality_relation_l936_93682

theorem quadratic_inequality_relation :
  (∀ x : ℝ, x > 3 → x^2 - 2*x - 3 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ ¬(x > 3)) :=
by sorry

end quadratic_inequality_relation_l936_93682


namespace grocer_coffee_solution_l936_93666

/-- Represents the grocer's coffee inventory --/
structure CoffeeInventory where
  initial : ℝ
  decafRatio : ℝ
  newPurchase : ℝ
  newDecafRatio : ℝ
  finalDecafRatio : ℝ

/-- The grocer's coffee inventory problem --/
def grocerProblem : CoffeeInventory where
  initial := 400  -- This is what we want to prove
  decafRatio := 0.2
  newPurchase := 100
  newDecafRatio := 0.5
  finalDecafRatio := 0.26

/-- Theorem stating the solution to the grocer's coffee inventory problem --/
theorem grocer_coffee_solution (inv : CoffeeInventory) : 
  inv.initial = 400 ∧ 
  inv.decafRatio = 0.2 ∧ 
  inv.newPurchase = 100 ∧ 
  inv.newDecafRatio = 0.5 ∧ 
  inv.finalDecafRatio = 0.26 →
  inv.finalDecafRatio * (inv.initial + inv.newPurchase) = 
    inv.decafRatio * inv.initial + inv.newDecafRatio * inv.newPurchase := by
  sorry

#check grocer_coffee_solution grocerProblem

end grocer_coffee_solution_l936_93666


namespace system_inequality_equivalence_l936_93625

theorem system_inequality_equivalence (x y m : ℝ) :
  (x - 2*y = 1 ∧ 2*x + y = 4*m) → (x + 3*y < 6 ↔ m < 7/4) := by
  sorry

end system_inequality_equivalence_l936_93625


namespace function_property_l936_93608

def f (a : ℝ) (x : ℝ) : ℝ := sorry

theorem function_property (a : ℝ) :
  (∀ x, f a (x + 3) = 3 * f a x) →
  (∀ x ∈ Set.Ioo 0 3, f a x = Real.log x - a * x) →
  a > 1/3 →
  (∃ x ∈ Set.Ioo (-6) (-3), f a x = -1/9 ∧ ∀ y ∈ Set.Ioo (-6) (-3), f a y ≤ f a x) →
  a = 1 := by sorry

end function_property_l936_93608


namespace fourth_number_in_row_15_l936_93664

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

theorem fourth_number_in_row_15 : pascal_triangle 15 3 = 455 := by
  sorry

end fourth_number_in_row_15_l936_93664


namespace contact_lenses_sales_l936_93621

/-- Proves that the total number of pairs of contact lenses sold is 11 given the problem conditions --/
theorem contact_lenses_sales (soft_price hard_price : ℕ) (soft_hard_diff total_sales : ℕ) :
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_diff = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ),
    soft = hard + soft_hard_diff ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 := by
  sorry

end contact_lenses_sales_l936_93621


namespace solve_inequality_find_m_range_l936_93641

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for part (2)
theorem find_m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end solve_inequality_find_m_range_l936_93641


namespace set_relationship_l936_93685

-- Define the sets M, P, and S
def M : Set ℤ := {x | ∃ k : ℤ, x = 3*k - 2}
def P : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 1}
def S : Set ℤ := {z | ∃ m : ℤ, z = 6*m + 1}

-- State the theorem
theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end set_relationship_l936_93685


namespace complex_equation_solution_l936_93615

theorem complex_equation_solution (a : ℝ) :
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end complex_equation_solution_l936_93615


namespace point_on_unit_circle_l936_93661

theorem point_on_unit_circle (s : ℝ) : 
  let x := (3 - s^2) / (3 + s^2)
  let y := 4*s / (3 + s^2)
  x^2 + y^2 = 1 := by
sorry

end point_on_unit_circle_l936_93661


namespace sqrt_x_minus_2_meaningful_l936_93616

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_meaningful_l936_93616


namespace opposite_of_sqrt_two_l936_93640

theorem opposite_of_sqrt_two : -(Real.sqrt 2) = -Real.sqrt 2 := by
  sorry

end opposite_of_sqrt_two_l936_93640


namespace coefficient_x4_in_expansion_l936_93668

theorem coefficient_x4_in_expansion : 
  (Finset.range 8).sum (fun k => Nat.choose 7 k * (1^(7-k) * x^(2*k))) = 
  21 * x^4 + (Finset.range 8).sum (fun k => if k ≠ 2 then Nat.choose 7 k * (1^(7-k) * x^(2*k)) else 0) := by
  sorry

end coefficient_x4_in_expansion_l936_93668


namespace prob_at_least_one_woman_l936_93645

/-- The probability of selecting at least one woman when choosing 4 people at random
    from a group of 8 men and 4 women is equal to 85/99. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (men.choose selected : ℚ) / (total.choose selected : ℚ) = 85 / 99 := by
  sorry

#check prob_at_least_one_woman

end prob_at_least_one_woman_l936_93645


namespace complex_modulus_l936_93697

theorem complex_modulus (Z : ℂ) (h : Z * Complex.I = 2 + Complex.I) : Complex.abs Z = Real.sqrt 5 := by
  sorry

end complex_modulus_l936_93697


namespace a_equals_one_l936_93600

def star (x y : ℝ) : ℝ := x + y - x * y

theorem a_equals_one (a : ℝ) (h : a = star 1 (star 0 1)) : a = 1 := by
  sorry

end a_equals_one_l936_93600


namespace pyramid_base_side_length_l936_93696

/-- Given a right pyramid with a square base, if the area of one lateral face
    is 120 square meters and the slant height is 40 meters, then the length
    of the side of its base is 6 meters. -/
theorem pyramid_base_side_length
  (area : ℝ) (slant_height : ℝ) (base_side : ℝ) :
  area = 120 →
  slant_height = 40 →
  area = (1/2) * base_side * slant_height →
  base_side = 6 :=
by sorry

end pyramid_base_side_length_l936_93696


namespace solution_set_inequality_l936_93630

theorem solution_set_inequality (x : ℝ) :
  x^2 - |x| - 2 ≤ 0 ↔ x ∈ Set.Icc (-2) 2 := by
  sorry

end solution_set_inequality_l936_93630


namespace quadratic_three_times_point_range_l936_93601

/-- A quadratic function y = -x^2 - x + c has at least one "three times point" (y = 3x) 
    in the range -3 < x < 1 if and only if -4 ≤ c < 5 -/
theorem quadratic_three_times_point_range (c : ℝ) : 
  (∃ x : ℝ, -3 < x ∧ x < 1 ∧ 3 * x = -x^2 - x + c) ↔ -4 ≤ c ∧ c < 5 :=
by sorry

end quadratic_three_times_point_range_l936_93601


namespace percentage_of_360_equals_144_l936_93633

theorem percentage_of_360_equals_144 : ∃ (p : ℚ), p * 360 = 144 ∧ p = 40 / 100 := by
  sorry

end percentage_of_360_equals_144_l936_93633


namespace circle_equation_l936_93626

/-- A circle C in the polar coordinate system -/
structure PolarCircle where
  /-- The point through which the circle passes -/
  passingPoint : (ℝ × ℝ)
  /-- The line equation whose intersection with polar axis determines the circle's center -/
  centerLine : ℝ → ℝ → Prop

/-- The polar equation of a circle -/
def polarEquation (c : PolarCircle) (ρ θ : ℝ) : Prop := sorry

theorem circle_equation (c : PolarCircle) (h1 : c.passingPoint = (Real.sqrt 2, π/4)) 
  (h2 : c.centerLine = fun ρ θ ↦ ρ * Real.sin (θ - π/3) = -Real.sqrt 3/2) :
  polarEquation c = fun ρ θ ↦ ρ = 2 * Real.cos θ := by sorry

end circle_equation_l936_93626


namespace average_age_is_35_l936_93636

/-- The average age of Omi, Kimiko, and Arlette is 35 years old. -/
theorem average_age_is_35 (kimiko_age omi_age arlette_age : ℕ) : 
  kimiko_age = 28 →
  omi_age = 2 * kimiko_age →
  arlette_age = 3 * kimiko_age / 4 →
  (omi_age + kimiko_age + arlette_age) / 3 = 35 := by
  sorry

end average_age_is_35_l936_93636


namespace complex_power_modulus_l936_93687

theorem complex_power_modulus : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by sorry

end complex_power_modulus_l936_93687


namespace probability_of_specific_draw_l936_93693

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Nat := 52

/-- Number of 5's in a standard deck -/
def NumberOfFives : Nat := 4

/-- Number of hearts in a standard deck -/
def NumberOfHearts : Nat := 13

/-- Number of Aces in a standard deck -/
def NumberOfAces : Nat := 4

/-- Probability of drawing a 5 as the first card, a heart as the second card, 
    and an Ace as the third card from a standard 52-card deck -/
def probabilityOfSpecificDraw : ℚ :=
  (NumberOfFives * NumberOfHearts * NumberOfAces) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_of_specific_draw :
  probabilityOfSpecificDraw = 1 / 663 := by
  sorry

end probability_of_specific_draw_l936_93693


namespace martha_cakes_l936_93622

/-- The number of whole cakes Martha needs to buy -/
def cakes_needed (num_children : ℕ) (cakes_per_child : ℕ) (special_children : ℕ) 
  (parts_per_cake : ℕ) : ℕ :=
  let total_small_cakes := num_children * cakes_per_child
  let special_whole_cakes := (special_children * cakes_per_child + parts_per_cake - 1) / parts_per_cake
  let remaining_small_cakes := total_small_cakes - special_whole_cakes * parts_per_cake
  special_whole_cakes + (remaining_small_cakes + parts_per_cake - 1) / parts_per_cake

/-- The theorem stating the number of cakes Martha needs to buy -/
theorem martha_cakes : cakes_needed 5 25 2 3 = 42 := by
  sorry

end martha_cakes_l936_93622


namespace shared_vertex_angle_is_84_l936_93672

/-- The angle between an edge of an equilateral triangle and an edge of a regular pentagon,
    when both shapes are inscribed in a circle and share a common vertex. -/
def shared_vertex_angle : ℝ := 84

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle :=
  (vertices : Fin 3 → ℝ × ℝ)
  (is_equilateral : ∀ i j : Fin 3, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 3, dist (vertices i) center = radius)

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle :=
  (vertices : Fin 5 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5)))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 5, dist (vertices i) center = radius)

theorem shared_vertex_angle_is_84 
  (triangle : EquilateralTriangleInCircle) 
  (pentagon : RegularPentagonInCircle) 
  (shared_vertex : ∃ i j, triangle.vertices i = pentagon.vertices j) :
  shared_vertex_angle = 84 := by
  sorry

end shared_vertex_angle_is_84_l936_93672


namespace grace_garden_seeds_l936_93699

/-- Represents the number of large beds in Grace's garden -/
def num_large_beds : Nat := 2

/-- Represents the number of medium beds in Grace's garden -/
def num_medium_beds : Nat := 2

/-- Represents the number of rows in a large bed -/
def rows_large_bed : Nat := 4

/-- Represents the number of rows in a medium bed -/
def rows_medium_bed : Nat := 3

/-- Represents the number of seeds per row in a large bed -/
def seeds_per_row_large : Nat := 25

/-- Represents the number of seeds per row in a medium bed -/
def seeds_per_row_medium : Nat := 20

/-- Calculates the total number of seeds Grace can plant in her raised bed garden -/
def total_seeds : Nat :=
  num_large_beds * rows_large_bed * seeds_per_row_large +
  num_medium_beds * rows_medium_bed * seeds_per_row_medium

/-- Proves that the total number of seeds Grace can plant is 320 -/
theorem grace_garden_seeds : total_seeds = 320 := by
  sorry

end grace_garden_seeds_l936_93699


namespace maria_savings_percentage_l936_93684

/-- Represents the "sundown deal" discount structure -/
structure SundownDeal where
  regular_price : ℝ
  second_pair_discount : ℝ
  additional_pair_discount : ℝ

/-- Calculates the total cost and savings for a given number of pairs -/
def calculate_deal (deal : SundownDeal) (num_pairs : ℕ) : ℝ × ℝ :=
  let regular_total := deal.regular_price * num_pairs
  let discounted_total := 
    if num_pairs ≥ 1 then deal.regular_price else 0 +
    if num_pairs ≥ 2 then deal.regular_price * (1 - deal.second_pair_discount) else 0 +
    if num_pairs > 2 then deal.regular_price * (1 - deal.additional_pair_discount) * (num_pairs - 2) else 0
  let savings := regular_total - discounted_total
  (discounted_total, savings)

/-- Theorem stating that Maria's savings percentage is 42% -/
theorem maria_savings_percentage (deal : SundownDeal) 
  (h1 : deal.regular_price = 60)
  (h2 : deal.second_pair_discount = 0.3)
  (h3 : deal.additional_pair_discount = 0.6) :
  let (_, savings) := calculate_deal deal 5
  let regular_total := deal.regular_price * 5
  (savings / regular_total) * 100 = 42 := by
  sorry


end maria_savings_percentage_l936_93684


namespace expression_value_l936_93655

theorem expression_value (x y : ℝ) (h : x - 2*y + 2 = 0) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 7 := by sorry

end expression_value_l936_93655


namespace triangle_equilateral_iff_sum_squares_eq_sum_products_l936_93658

/-- A triangle with sides a, b, and c is equilateral if and only if a² + b² + c² = ab + bc + ca -/
theorem triangle_equilateral_iff_sum_squares_eq_sum_products {a b c : ℝ} (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a = b ∧ b = c) ↔ a^2 + b^2 + c^2 = a*b + b*c + c*a := by sorry

end triangle_equilateral_iff_sum_squares_eq_sum_products_l936_93658


namespace leadership_assignment_theorem_l936_93624

def community_size : ℕ := 12
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def senior_officer_count : ℕ := 2
def inferior_officer_count : ℕ := 2

def leadership_assignment_count : ℕ :=
  community_size *
  (community_size - chief_count).choose supporting_chief_count *
  (community_size - chief_count - supporting_chief_count).choose senior_officer_count *
  (community_size - chief_count - supporting_chief_count - senior_officer_count).choose inferior_officer_count

theorem leadership_assignment_theorem :
  leadership_assignment_count = 498960 := by
  sorry

end leadership_assignment_theorem_l936_93624


namespace kelly_textbook_weight_difference_l936_93680

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem kelly_textbook_weight_difference :
  let chemistry_weight : ℚ := 7125 / 1000
  let geometry_weight : ℚ := 625 / 1000
  chemistry_weight - geometry_weight = 13 / 2 := by
  sorry

end kelly_textbook_weight_difference_l936_93680


namespace max_students_distribution_l936_93643

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
  sorry

end max_students_distribution_l936_93643


namespace gcd_63_84_l936_93609

theorem gcd_63_84 : Nat.gcd 63 84 = 21 := by
  sorry

end gcd_63_84_l936_93609


namespace matrix_operation_result_l936_93642

theorem matrix_operation_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 0, -3]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![6, 0; -1, 8]
  A - B + C = !![12, -7; 1, 16] := by
sorry

end matrix_operation_result_l936_93642


namespace dunkers_lineup_count_l936_93681

theorem dunkers_lineup_count (n : ℕ) (k : ℕ) (a : ℕ) (z : ℕ) : 
  n = 15 → k = 5 → a ≠ z → a ≤ n → z ≤ n →
  (Nat.choose (n - 2) (k - 1) * 2 + Nat.choose (n - 2) k) = 2717 :=
by sorry

end dunkers_lineup_count_l936_93681


namespace quadratic_solution_property_l936_93651

theorem quadratic_solution_property (a b : ℝ) :
  (a * 1^2 + b * 1 - 1 = 0) → (2023 - a - b = 2022) := by
  sorry

end quadratic_solution_property_l936_93651


namespace intersection_complement_A_and_B_l936_93604

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f : A → ℝ := fun x ↦ 2 - x

-- Define the range of f as B
def B : Set ℝ := Set.range f

-- Theorem statement
theorem intersection_complement_A_and_B :
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 ∪ {2} :=
sorry

end intersection_complement_A_and_B_l936_93604


namespace symmetric_point_wrt_y_axis_l936_93623

/-- Given a point A(3,1) in a Cartesian coordinate system, 
    its symmetric point with respect to the y-axis has coordinates (-3,1). -/
theorem symmetric_point_wrt_y_axis : 
  let A : ℝ × ℝ := (3, 1)
  let symmetric_point := (-A.1, A.2)
  symmetric_point = (-3, 1) := by sorry

end symmetric_point_wrt_y_axis_l936_93623


namespace contradiction_proof_l936_93605

theorem contradiction_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : x + y > 0 := by
  sorry

end contradiction_proof_l936_93605


namespace geometric_sequence_max_value_l936_93656

/-- Given a geometric sequence {a_n} with common ratio √2, 
    T_n = (17S_n - S_{2n}) / a_{n+1} attains its maximum value when n = 4 -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →
  (∀ n, S n = a 1 * (1 - (Real.sqrt 2)^n) / (1 - Real.sqrt 2)) →
  (∀ n, T n = (17 * S n - S (2 * n)) / a (n + 1)) →
  (∃ B : ℝ, ∀ n, T n ≤ B ∧ T 4 = B) :=
by sorry

end geometric_sequence_max_value_l936_93656


namespace jessica_fraction_proof_l936_93670

/-- Represents Jessica's collection of quarters -/
structure QuarterCollection where
  total : ℕ
  from_1790s : ℕ

/-- The fraction of quarters from states admitted in 1790-1799 -/
def fraction_from_1790s (c : QuarterCollection) : ℚ :=
  c.from_1790s / c.total

/-- Jessica's actual collection -/
def jessica_collection : QuarterCollection :=
  { total := 30, from_1790s := 16 }

theorem jessica_fraction_proof :
  fraction_from_1790s jessica_collection = 8 / 15 := by
  sorry

end jessica_fraction_proof_l936_93670


namespace angle_expression_value_l936_93612

theorem angle_expression_value (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin α = Real.sqrt 15 / 4) :  -- sin α = √15/4
  Real.sin (α + π/4) / (Real.sin (2*α) + Real.cos (2*α) + 1) = -Real.sqrt 2 := by
  sorry

end angle_expression_value_l936_93612


namespace exactly_one_statement_correct_l936_93662

-- Define rational and irrational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Define the four statements
def Statement1 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i)

def Statement2 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r * i)

def Statement3 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ + i₂)

def Statement4 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ * i₂)

-- The main theorem
theorem exactly_one_statement_correct :
  (Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ ¬Statement4) :=
sorry

end exactly_one_statement_correct_l936_93662


namespace arithmetic_sum_equals_180_l936_93628

/-- The sum of an arithmetic sequence with first term 30, common difference 10, and 4 terms -/
def arithmeticSum : ℕ := sorry

/-- The first term of the sequence -/
def firstTerm : ℕ := 30

/-- The common difference between consecutive terms -/
def commonDifference : ℕ := 10

/-- The number of terms in the sequence -/
def numberOfTerms : ℕ := 4

/-- Theorem stating that the sum of the arithmetic sequence is 180 -/
theorem arithmetic_sum_equals_180 : arithmeticSum = 180 := by sorry

end arithmetic_sum_equals_180_l936_93628


namespace point_in_fourth_quadrant_l936_93675

def i : ℂ := Complex.I

theorem point_in_fourth_quadrant :
  let z : ℂ := (5 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end point_in_fourth_quadrant_l936_93675


namespace hyperbola_sum_l936_93698

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  (3 + Real.sqrt 45 - 3)^2 = c^2 ∧ 
  (6 - 3)^2 = a^2 ∧ 
  b^2 = c^2 - a^2 → 
  h + k + a + b = 11 := by
sorry

end hyperbola_sum_l936_93698


namespace least_integer_satisfying_inequality_l936_93695

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), 3 * |y| - 2 * y + 8 < 23 → x ≤ y) ∧ (3 * |x| - 2 * x + 8 < 23) :=
by sorry

end least_integer_satisfying_inequality_l936_93695


namespace contrapositive_equivalence_l936_93614

theorem contrapositive_equivalence (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) := by
  sorry

end contrapositive_equivalence_l936_93614


namespace shoe_multiple_l936_93683

theorem shoe_multiple (jacob edward brian : ℕ) : 
  jacob = edward / 2 →
  brian = 22 →
  jacob + edward + brian = 121 →
  edward / brian = 3 :=
by sorry

end shoe_multiple_l936_93683


namespace fraction_equality_l936_93620

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 →
    (P / (x + 3) + Q / (x^2 - 10*x + 16) = (x^2 - 6*x + 18) / (x^3 - 7*x^2 + 14*x - 48))) →
  (Q : ℚ) / P = 10 / 3 := by
sorry

end fraction_equality_l936_93620


namespace arithmetic_mean_problem_l936_93638

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + (3 * x) + 15 + (3 * x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end arithmetic_mean_problem_l936_93638


namespace pet_insurance_coverage_percentage_l936_93639

theorem pet_insurance_coverage_percentage
  (insurance_duration : ℕ)
  (insurance_monthly_cost : ℚ)
  (procedure_cost : ℚ)
  (amount_saved : ℚ)
  (h1 : insurance_duration = 24)
  (h2 : insurance_monthly_cost = 20)
  (h3 : procedure_cost = 5000)
  (h4 : amount_saved = 3520)
  : (1 - (amount_saved / procedure_cost)) * 100 = 20 := by
  sorry

end pet_insurance_coverage_percentage_l936_93639


namespace rectangle_area_l936_93692

theorem rectangle_area (l w : ℕ) : 
  l * l + w * w = 17 * 17 →  -- diagonal is 17 cm
  2 * l + 2 * w = 46 →       -- perimeter is 46 cm
  l * w = 120 :=             -- area is 120 cm²
by
  sorry

end rectangle_area_l936_93692


namespace checkers_game_possibilities_l936_93671

/-- Represents the number of games played by each friend in a checkers game. -/
structure CheckersGames where
  friend1 : ℕ
  friend2 : ℕ
  friend3 : ℕ

/-- Checks if the given number of games for three friends is valid. -/
def isValidGameCount (games : CheckersGames) : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = (games.friend1 + games.friend2 + games.friend3) / 2 ∧
    a + c = games.friend1 ∧
    b + c = games.friend2 ∧
    a + b = games.friend3

/-- Theorem stating the validity of different game counts for the third friend. -/
theorem checkers_game_possibilities : 
  let games1 := CheckersGames.mk 25 17 34
  let games2 := CheckersGames.mk 25 17 35
  let games3 := CheckersGames.mk 25 17 56
  isValidGameCount games1 ∧ 
  ¬isValidGameCount games2 ∧ 
  ¬isValidGameCount games3 := by
  sorry

end checkers_game_possibilities_l936_93671


namespace circle_diameter_from_area_l936_93613

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 := by
  sorry

end circle_diameter_from_area_l936_93613


namespace no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l936_93602

theorem no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four 
  (p : Nat) (hp : Prime p) (hp_cong : p % 4 = 3) :
  ∀ n : Nat, n > 0 → ¬∃ x y : Nat, x > 0 ∧ y > 0 ∧ x^2 + y^2 = p^n :=
by sorry

end no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l936_93602


namespace pencil_notebook_cost_l936_93634

/-- Given the cost of pencils and notebooks, calculate the cost of a different quantity -/
theorem pencil_notebook_cost 
  (pencil_price notebook_price : ℕ) 
  (h1 : 4 * pencil_price + 3 * notebook_price = 9600)
  (h2 : 2 * pencil_price + 2 * notebook_price = 5400) :
  8 * pencil_price + 7 * notebook_price = 20400 :=
by sorry

end pencil_notebook_cost_l936_93634


namespace choose_three_from_nine_l936_93677

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l936_93677


namespace least_three_digit_multiple_l936_93606

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  n = 168 :=
by sorry

end least_three_digit_multiple_l936_93606


namespace polynomial_difference_simplification_l936_93663

/-- The difference of two polynomials is equal to a simplified polynomial. -/
theorem polynomial_difference_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x^2 + 7) - 
  (x^6 + 4 * x^5 + 2 * x^4 - x^3 + x^2 + 8) = 
  x^6 - x^5 - x^4 + 6 * x^3 - 1 := by
  sorry

end polynomial_difference_simplification_l936_93663


namespace partner_q_invest_time_l936_93619

/-- Represents the investment and profit data for three partners -/
structure PartnerData where
  investment_ratio : Fin 3 → ℚ
  profit_ratio : Fin 3 → ℚ
  p_invest_time : ℚ
  r_invest_time : ℚ

/-- Calculates the investment time for partner q given the partner data -/
def calculate_q_invest_time (data : PartnerData) : ℚ :=
  (data.investment_ratio 0 * data.p_invest_time * data.profit_ratio 1) /
  (data.investment_ratio 1 * data.profit_ratio 0)

/-- Theorem stating that partner q's investment time is 14 months -/
theorem partner_q_invest_time (data : PartnerData)
  (h1 : data.investment_ratio 0 = 7)
  (h2 : data.investment_ratio 1 = 5)
  (h3 : data.investment_ratio 2 = 3)
  (h4 : data.profit_ratio 0 = 7)
  (h5 : data.profit_ratio 1 = 14)
  (h6 : data.profit_ratio 2 = 9)
  (h7 : data.p_invest_time = 5)
  (h8 : data.r_invest_time = 9) :
  calculate_q_invest_time data = 14 := by
  sorry

end partner_q_invest_time_l936_93619


namespace permutations_without_patterns_l936_93667

/-- The total number of permutations of 4 x's, 3 y's, and 2 z's -/
def total_permutations : ℕ := 1260

/-- The set of permutations where the pattern xxxx appears -/
def A₁ : Finset (List Char) := sorry

/-- The set of permutations where the pattern yyy appears -/
def A₂ : Finset (List Char) := sorry

/-- The set of permutations where the pattern zz appears -/
def A₃ : Finset (List Char) := sorry

/-- The theorem to be proved -/
theorem permutations_without_patterns (h₁ : Finset.card A₁ = 60) 
  (h₂ : Finset.card A₂ = 105) (h₃ : Finset.card A₃ = 280)
  (h₄ : Finset.card (A₁ ∩ A₂) = 12) (h₅ : Finset.card (A₁ ∩ A₃) = 20)
  (h₆ : Finset.card (A₂ ∩ A₃) = 30) (h₇ : Finset.card (A₁ ∩ A₂ ∩ A₃) = 6) :
  total_permutations - Finset.card (A₁ ∪ A₂ ∪ A₃) = 871 := by
  sorry

end permutations_without_patterns_l936_93667


namespace jacob_lunch_calories_l936_93654

theorem jacob_lunch_calories (planned : ℕ) (breakfast dinner extra : ℕ) 
  (h1 : planned < 1800)
  (h2 : breakfast = 400)
  (h3 : dinner = 1100)
  (h4 : extra = 600) :
  planned + extra - (breakfast + dinner) = 900 :=
by sorry

end jacob_lunch_calories_l936_93654


namespace multiple_calculation_l936_93635

theorem multiple_calculation (a b m : ℤ) : 
  b = 8 → 
  b - a = 3 → 
  a * b = m * (a + b) + 14 → 
  m = 2 := by
sorry

end multiple_calculation_l936_93635


namespace dz_dt_formula_l936_93673

noncomputable def z (t : ℝ) := Real.arcsin ((2*t)^2 + (4*t^2)^2 + t^2)

theorem dz_dt_formula (t : ℝ) :
  deriv z t = (2*t*(1 + 4*t + 32*t^2)) / Real.sqrt (1 - ((2*t)^2 + (4*t^2)^2 + t^2)^2) :=
sorry

end dz_dt_formula_l936_93673


namespace gems_calculation_l936_93631

/-- Calculates the total number of gems received given an initial spend, gem rate, and bonus percentage. -/
def total_gems (spend : ℕ) (rate : ℕ) (bonus_percent : ℕ) : ℕ :=
  let initial_gems := spend * rate
  let bonus_gems := initial_gems * bonus_percent / 100
  initial_gems + bonus_gems

/-- Proves that given the specified conditions, the total number of gems received is 30000. -/
theorem gems_calculation :
  let spend := 250
  let rate := 100
  let bonus_percent := 20
  total_gems spend rate bonus_percent = 30000 := by
  sorry

end gems_calculation_l936_93631


namespace cone_surface_area_l936_93610

theorem cone_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 2 * Real.sqrt 5) :
  let slant_height := Real.sqrt (r^2 + h^2)
  let base_area := π * r^2
  let lateral_area := π * r * slant_height
  base_area + lateral_area = 40 * π :=
by sorry

end cone_surface_area_l936_93610


namespace rook_placements_corners_removed_8x8_l936_93649

/-- Represents a chessboard with corners removed -/
def CornersRemovedChessboard : Type := Unit

/-- The number of ways to place non-attacking rooks on a corners-removed chessboard -/
def num_rook_placements (board : CornersRemovedChessboard) : ℕ := 21600

/-- The theorem stating the number of ways to place eight non-attacking rooks
    on an 8x8 chessboard with its four corners removed -/
theorem rook_placements_corners_removed_8x8 (board : CornersRemovedChessboard) :
  num_rook_placements board = 21600 := by sorry

end rook_placements_corners_removed_8x8_l936_93649


namespace no_integer_square_root_l936_93659

theorem no_integer_square_root : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1 = y^2 := by
  sorry

end no_integer_square_root_l936_93659


namespace no_valid_coloring_l936_93644

/-- A coloring function that assigns one of three colors to each natural number -/
def Coloring := ℕ → Fin 3

/-- Predicate checking if a coloring satisfies the required property -/
def ValidColoring (c : Coloring) : Prop :=
  (∃ n : ℕ, c n = 0) ∧
  (∃ n : ℕ, c n = 1) ∧
  (∃ n : ℕ, c n = 2) ∧
  (∀ x y : ℕ, c x ≠ c y → c (x + y) ≠ c x ∧ c (x + y) ≠ c y)

theorem no_valid_coloring : ¬∃ c : Coloring, ValidColoring c := by
  sorry

end no_valid_coloring_l936_93644


namespace book_price_increase_l936_93648

theorem book_price_increase (initial_price : ℝ) : 
  let decrease_rate : ℝ := 0.20
  let net_change_rate : ℝ := 0.11999999999999986
  let price_after_decrease : ℝ := initial_price * (1 - decrease_rate)
  let final_price : ℝ := initial_price * (1 + net_change_rate)
  ∃ (increase_rate : ℝ), 
    price_after_decrease * (1 + increase_rate) = final_price ∧ 
    abs (increase_rate - 0.4) < 0.00000000000001 := by
  sorry

end book_price_increase_l936_93648


namespace xyz_range_l936_93689

theorem xyz_range (x y z : ℝ) 
  (sum_condition : x + y + z = 1) 
  (square_sum_condition : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5/27 := by
  sorry

end xyz_range_l936_93689


namespace cubic_polynomial_problem_l936_93632

variable (a b c : ℝ)
variable (P : ℝ → ℝ)

theorem cubic_polynomial_problem :
  (∀ x, x^3 - 2*x^2 - 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + 2*c →
  P b = 2*a + c →
  P c = a + 2*b →
  P (a + b + c) = -20 →
  ∀ x, P x = 4*x^3 - 6*x^2 - 12*x := by
sorry

end cubic_polynomial_problem_l936_93632


namespace product_of_sums_l936_93652

theorem product_of_sums (x y : ℝ) (h1 : x + y = -3) (h2 : x * y = 1) :
  (x + 5) * (y + 5) = 11 := by
  sorry

end product_of_sums_l936_93652


namespace convention_handshakes_l936_93686

theorem convention_handshakes (twin_sets triplet_sets : ℕ) 
  (h1 : twin_sets = 10)
  (h2 : triplet_sets = 7)
  (h3 : ∀ t : ℕ, t ≤ twin_sets → (t * 2 - 2) * 2 = t * 2 * (t * 2 - 2))
  (h4 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3 - 3) * 3 = t * 3 * (t * 3 - 3))
  (h5 : ∀ t : ℕ, t ≤ twin_sets → (t * 2) * (2 * triplet_sets) = 3 * (t * 2) * triplet_sets)
  (h6 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3) * (2 * twin_sets) = 3 * (t * 3) * twin_sets) :
  ((twin_sets * 2) * ((twin_sets * 2) - 2)) / 2 +
  ((triplet_sets * 3) * ((triplet_sets * 3) - 3)) / 2 +
  (twin_sets * 2) * (2 * triplet_sets) / 3 +
  (triplet_sets * 3) * (2 * twin_sets) / 3 = 922 := by
sorry

end convention_handshakes_l936_93686


namespace total_revenue_proof_l936_93676

def sneakers_price : ℝ := 80
def sandals_price : ℝ := 60
def boots_price : ℝ := 120

def sneakers_discount : ℝ := 0.25
def sandals_discount : ℝ := 0.35
def boots_discount : ℝ := 0.40

def sneakers_quantity : ℕ := 2
def sandals_quantity : ℕ := 4
def boots_quantity : ℕ := 11

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def revenue (price discount quantity : ℝ) : ℝ :=
  discounted_price price discount * quantity

theorem total_revenue_proof :
  revenue sneakers_price sneakers_discount (sneakers_quantity : ℝ) +
  revenue sandals_price sandals_discount (sandals_quantity : ℝ) +
  revenue boots_price boots_discount (boots_quantity : ℝ) = 1068 := by
  sorry

end total_revenue_proof_l936_93676


namespace dawsons_friends_l936_93660

def total_cost : ℕ := 13500
def cost_per_person : ℕ := 900

theorem dawsons_friends :
  (total_cost / cost_per_person) - 1 = 14 := by
  sorry

end dawsons_friends_l936_93660


namespace trigonometric_inequalities_l936_93691

theorem trigonometric_inequalities :
  (Real.tan (3 * π / 5) < Real.tan (π / 5)) ∧
  (Real.cos (-17 * π / 4) > Real.cos (-23 * π / 5)) := by
  sorry

end trigonometric_inequalities_l936_93691


namespace quadratic_root_and_m_l936_93679

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x + m = 0

-- Theorem statement
theorem quadratic_root_and_m :
  ∀ m : ℝ, quadratic_equation (-2) m → m = 0 ∧ quadratic_equation 0 m :=
by
  sorry

end quadratic_root_and_m_l936_93679


namespace volume_equality_l936_93627

/-- The region R₁ bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def R₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region R₂ satisfying x² - y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def R₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The volume V₁ obtained by rotating R₁ about the y-axis -/
noncomputable def V₁ : ℝ := sorry

/-- The volume V₂ obtained by rotating R₂ about the y-axis -/
noncomputable def V₂ : ℝ := sorry

/-- The theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end volume_equality_l936_93627


namespace mary_sugar_amount_l936_93665

/-- The amount of sugar required by the recipe in cups -/
def total_sugar : ℕ := 14

/-- The amount of sugar Mary still needs to add in cups -/
def sugar_to_add : ℕ := 12

/-- The amount of sugar Mary has already put in -/
def sugar_already_added : ℕ := total_sugar - sugar_to_add

theorem mary_sugar_amount : sugar_already_added = 2 := by
  sorry

end mary_sugar_amount_l936_93665


namespace quadratic_always_positive_l936_93694

/-- A quadratic function is always positive if and only if its coefficient of x^2 is positive and its discriminant is negative -/
theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_always_positive_l936_93694


namespace tails_appearance_l936_93617

/-- The number of coin flips -/
def total_flips : ℕ := 20

/-- The frequency of getting "heads" -/
def heads_frequency : ℚ := 45/100

/-- The number of times "tails" appears -/
def tails_count : ℕ := 11

/-- Theorem: Given a coin flipped 20 times with a frequency of getting "heads" of 0.45,
    the number of times "tails" appears is 11. -/
theorem tails_appearance :
  (total_flips : ℚ) * (1 - heads_frequency) = tails_count := by sorry

end tails_appearance_l936_93617


namespace store_a_cheaper_l936_93607

/-- Represents the cost function for Store A -/
def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 10 + 0.7 * (x - 10)

/-- Represents the cost function for Store B -/
def cost_store_b (x : ℕ) : ℝ := 0.85 * x

/-- The number of exercise books Xiao Ming wants to buy -/
def num_books : ℕ := 22

theorem store_a_cheaper :
  cost_store_a num_books < cost_store_b num_books :=
sorry

end store_a_cheaper_l936_93607


namespace problem_l936_93690

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem problem (n : ℕ) (d : ℕ → ℕ) :
  n > 0 ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 15 → d i < d j) ∧
  (∀ i, 1 ≤ i ∧ i ≤ 15 → is_divisor (d i) n) ∧
  d 1 = 1 ∧
  n = d 13 + d 14 + d 15 ∧
  (d 5 + 1)^3 = d 15 + 1 →
  n = 1998 := by
sorry

end problem_l936_93690


namespace u_value_when_m_is_3_l936_93611

-- Define the functions u and t
def t (m : ℕ) : ℕ := 3^m + m
def u (m : ℕ) : ℕ := 4^(t m) - 3*(t m)

-- State the theorem
theorem u_value_when_m_is_3 : u 3 = 4^30 - 90 := by
  sorry

end u_value_when_m_is_3_l936_93611


namespace chocolate_bar_sales_l936_93646

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that selling 4 out of 11 bars at $4 each yields $16 -/
theorem chocolate_bar_sales : money_made 11 4 7 = 16 := by
  sorry

end chocolate_bar_sales_l936_93646


namespace complex_magnitude_l936_93647

theorem complex_magnitude (z : ℂ) : z = (1 + Complex.I) / (2 - 2 * Complex.I) → Complex.abs z = 1 / 2 := by
  sorry

end complex_magnitude_l936_93647


namespace hyperbola_vertex_distance_l936_93674

/-- The distance between the vertices of the hyperbola (x²/16) - (y²/25) = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => (x^2 / 16) - (y^2 / 25) = 1
  ∃ x₁ x₂ : ℝ, (h x₁ 0 ∧ h x₂ 0) ∧ |x₁ - x₂| = 8 :=
sorry

end hyperbola_vertex_distance_l936_93674


namespace base12_remainder_is_4_l936_93618

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 5, 3, 4]

/-- The theorem stating that the remainder of the base-12 number divided by 9 is 4 --/
theorem base12_remainder_is_4 : 
  (base12ToBase10 base12Number) % 9 = 4 := by
  sorry

end base12_remainder_is_4_l936_93618


namespace girl_multiplication_problem_l936_93657

theorem girl_multiplication_problem (mistake_factor : ℕ) (difference : ℕ) (base : ℕ) (correct_factor : ℕ) : 
  mistake_factor = 34 →
  difference = 1233 →
  base = 137 →
  base * correct_factor = base * mistake_factor + difference →
  correct_factor = 43 := by
sorry

end girl_multiplication_problem_l936_93657
