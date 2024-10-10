import Mathlib

namespace square_perimeter_l2313_231344

theorem square_perimeter (area : ℝ) (side : ℝ) : 
  area = 400 ∧ area = side * side → 4 * side = 80 := by
  sorry

end square_perimeter_l2313_231344


namespace inequality_solution_l2313_231360

theorem inequality_solution (x : ℝ) : 4 ≤ (2*x)/(3*x-7) ∧ (2*x)/(3*x-7) < 9 ↔ 63/25 < x ∧ x ≤ 2.8 := by
  sorry

end inequality_solution_l2313_231360


namespace expression_value_l2313_231318

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) :
  8 - 6 * a + 9 * b = 2 := by
  sorry

end expression_value_l2313_231318


namespace sqrt_product_equals_150_sqrt_3_l2313_231351

theorem sqrt_product_equals_150_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 45 * Real.sqrt 20 = 150 * Real.sqrt 3 := by
  sorry

end sqrt_product_equals_150_sqrt_3_l2313_231351


namespace lcm_1362_918_l2313_231369

theorem lcm_1362_918 : Nat.lcm 1362 918 = 69462 := by
  sorry

end lcm_1362_918_l2313_231369


namespace circle_equation_center_radius_l2313_231396

/-- Given a circle equation, prove its center and radius -/
theorem circle_equation_center_radius 
  (x y : ℝ) 
  (h : x^2 - 2*x + y^2 + 6*y = 6) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -3) ∧ 
    radius = 4 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end circle_equation_center_radius_l2313_231396


namespace equation_solution_l2313_231394

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_solution :
  let d : ℝ := 4
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - d) ∧ x = 8 := by
  sorry

end equation_solution_l2313_231394


namespace simplify_polynomial_l2313_231319

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^2 - 3*x + 1) - 4*(2*x^2 - 3*x + 5) = 8*x^3 - 14*x^2 + 14*x - 20 := by
  sorry

end simplify_polynomial_l2313_231319


namespace quadratic_inequality_boundary_l2313_231363

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 1) < c ↔ -5/2 < x ∧ x < 3) → c = 27 :=
by sorry

end quadratic_inequality_boundary_l2313_231363


namespace at_least_one_equation_has_solution_l2313_231356

theorem at_least_one_equation_has_solution (a b c : ℝ) : 
  ¬(c^2 > a^2 + b^2 ∧ b^2 - 16*a*c < 0) := by
sorry

end at_least_one_equation_has_solution_l2313_231356


namespace james_weight_vest_savings_l2313_231342

/-- The savings James makes by buying a separate vest and plates instead of a discounted 200-pound weight vest -/
theorem james_weight_vest_savings 
  (separate_vest_cost : ℝ) 
  (plate_weight : ℝ) 
  (cost_per_pound : ℝ) 
  (full_vest_cost : ℝ) 
  (discount : ℝ)
  (h1 : separate_vest_cost = 250)
  (h2 : plate_weight = 200)
  (h3 : cost_per_pound = 1.2)
  (h4 : full_vest_cost = 700)
  (h5 : discount = 100) :
  full_vest_cost - discount - (separate_vest_cost + plate_weight * cost_per_pound) = 110 := by
  sorry

end james_weight_vest_savings_l2313_231342


namespace pirate_loot_sum_l2313_231338

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement -/
theorem pirate_loot_sum : 
  let silverware := base5ToBase10 [4, 1, 2, 3]
  let gemstones := base5ToBase10 [2, 2, 0, 3]
  let fine_silk := base5ToBase10 [2, 0, 2]
  silverware + gemstones + fine_silk = 873 := by
  sorry


end pirate_loot_sum_l2313_231338


namespace equation_solution_l2313_231384

theorem equation_solution (x : ℝ) : 
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = 2 * Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 6 + 2 * k * Real.pi) :=
by sorry

end equation_solution_l2313_231384


namespace polynomial_factorization_l2313_231310

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b)^2 + (b + c)^2 + (c + a)^2) := by
  sorry

end polynomial_factorization_l2313_231310


namespace infinitely_many_linear_combinations_l2313_231333

/-- An infinite sequence of positive integers where each element is strictly greater than the previous one. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

/-- The property that an element of the sequence can be expressed as a linear combination of two distinct earlier elements. -/
def CanBeExpressedAsLinearCombination (a : ℕ → ℕ) (m p q : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

/-- The main theorem stating that infinitely many elements of the sequence can be expressed as linear combinations of two distinct earlier elements. -/
theorem infinitely_many_linear_combinations (a : ℕ → ℕ) 
    (h : StrictlyIncreasingSequence a) :
    ∀ N, ∃ m, m > N ∧ ∃ p q, CanBeExpressedAsLinearCombination a m p q := by
  sorry

end infinitely_many_linear_combinations_l2313_231333


namespace remainder_theorem_l2313_231307

/-- The polynomial f(x) = 4x^5 - 9x^4 + 7x^2 - x - 35 -/
def f (x : ℝ) : ℝ := 4 * x^5 - 9 * x^4 + 7 * x^2 - x - 35

/-- The theorem stating that the remainder when f(x) is divided by (x - 2.5) is 45.3125 -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = λ x => (x - 2.5) * q x + 45.3125 := by sorry

end remainder_theorem_l2313_231307


namespace triangle_theorem_l2313_231309

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A - t.c * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.a = 2 * Real.sqrt 3)
  (h3 : Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3) :
  Real.cos t.A = 1/3 ∧ t.c = 3 := by
  sorry


end triangle_theorem_l2313_231309


namespace constant_function_satisfies_inequality_l2313_231375

theorem constant_function_satisfies_inequality :
  ∀ f : ℕ → ℝ,
  (∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 → f (a * c) + f (b * c) - f c * f (a * b) ≥ 1) →
  (∀ x : ℕ, f x = 1) :=
by sorry

end constant_function_satisfies_inequality_l2313_231375


namespace classroom_problem_l2313_231362

/-- Calculates the final number of children in a classroom after some changes -/
def final_children_count (initial_boys initial_girls boys_left girls_entered : ℕ) : ℕ :=
  (initial_boys - boys_left) + (initial_girls + girls_entered)

/-- Proves that the final number of children in the classroom is 8 -/
theorem classroom_problem :
  let initial_boys : ℕ := 5
  let initial_girls : ℕ := 4
  let boys_left : ℕ := 3
  let girls_entered : ℕ := 2
  final_children_count initial_boys initial_girls boys_left girls_entered = 8 := by
  sorry

#eval final_children_count 5 4 3 2

end classroom_problem_l2313_231362


namespace pool_water_removal_l2313_231391

/-- Calculates the volume of water removed from a rectangular pool in gallons -/
def water_removed (length width height : ℝ) (conversion_factor : ℝ) : ℝ :=
  length * width * height * conversion_factor

theorem pool_water_removal :
  let length : ℝ := 60
  let width : ℝ := 10
  let height : ℝ := 0.5
  let conversion_factor : ℝ := 7.5
  water_removed length width height conversion_factor = 2250 := by
sorry

end pool_water_removal_l2313_231391


namespace women_in_room_l2313_231336

theorem women_in_room (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 5 = 23 →
  3 * (initial_women - 4) = 57 :=
by sorry

end women_in_room_l2313_231336


namespace grandfathers_age_l2313_231314

theorem grandfathers_age (grandfather_age : ℕ) (xiaoming_age : ℕ) : 
  grandfather_age > 7 * xiaoming_age →
  grandfather_age < 70 →
  ∃ (k : ℕ), grandfather_age - xiaoming_age = 60 * k →
  grandfather_age = 69 :=
by
  sorry

end grandfathers_age_l2313_231314


namespace polynomial_value_at_three_l2313_231346

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  x^6 - 6*x^2 + 7*x = 696 := by
sorry

end polynomial_value_at_three_l2313_231346


namespace angle_between_skew_medians_l2313_231386

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A median of a face in a regular tetrahedron -/
structure FaceMedian (t : RegularTetrahedron a) where
  start_vertex : ℝ × ℝ × ℝ
  end_point : ℝ × ℝ × ℝ

/-- The angle between two vectors in ℝ³ -/
def angle_between (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Two face medians are skew if they're not on the same face -/
def are_skew_medians (m1 m2 : FaceMedian t) : Prop := sorry

theorem angle_between_skew_medians (t : RegularTetrahedron a) 
  (m1 m2 : FaceMedian t) (h : are_skew_medians m1 m2) : 
  angle_between (m1.end_point - m1.start_vertex) (m2.end_point - m2.start_vertex) = Real.arccos (1/6) := by
  sorry

end angle_between_skew_medians_l2313_231386


namespace pigeonhole_principle_sports_l2313_231334

theorem pigeonhole_principle_sports (n : ℕ) (h : n = 50) :
  ∃ (same_choices : ℕ), same_choices ≥ 3 ∧
  (∀ (choices : Fin n → Fin 4 × Fin 3 × Fin 2),
   ∃ (subset : Finset (Fin n)),
   subset.card = same_choices ∧
   ∀ (i j : Fin n), i ∈ subset → j ∈ subset → choices i = choices j) :=
by sorry

end pigeonhole_principle_sports_l2313_231334


namespace scalene_triangle_unique_x_l2313_231313

/-- Represents a scalene triangle with specific properties -/
structure ScaleneTriangle where
  -- One angle is 45 degrees
  angle1 : ℝ
  angle1_eq : angle1 = 45
  -- Another angle is x degrees
  angle2 : ℝ
  -- The third angle
  angle3 : ℝ
  -- The sum of all angles is 180 degrees
  angle_sum : angle1 + angle2 + angle3 = 180
  -- The sides opposite angle1 and angle2 are equal
  equal_sides : True
  -- The triangle is scalene (all sides are different)
  is_scalene : True

/-- 
Theorem: In a scalene triangle with one angle of 45° and another angle of x°, 
where the side lengths opposite these two angles are equal, 
the only possible value for x is 45°.
-/
theorem scalene_triangle_unique_x (t : ScaleneTriangle) : t.angle2 = 45 := by
  sorry

#check scalene_triangle_unique_x

end scalene_triangle_unique_x_l2313_231313


namespace max_value_trig_sum_l2313_231376

open Real

theorem max_value_trig_sum (α β γ δ ε : ℝ) : 
  (∀ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α ≤ 5) ∧ 
  (∃ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α = 5) := by
  sorry

end max_value_trig_sum_l2313_231376


namespace remaining_amount_proof_l2313_231321

def initial_amount : ℕ := 87
def spent_amount : ℕ := 64

theorem remaining_amount_proof :
  initial_amount - spent_amount = 23 :=
by sorry

end remaining_amount_proof_l2313_231321


namespace fraction_comparison_l2313_231365

theorem fraction_comparison : (291 : ℚ) / 730 > 29 / 73 := by
  sorry

end fraction_comparison_l2313_231365


namespace original_price_correct_l2313_231331

/-- The original price of a bag of mini peanut butter cups before discount -/
def original_price : ℝ := 6

/-- The discount percentage applied to the bags -/
def discount_percentage : ℝ := 0.75

/-- The number of bags purchased -/
def num_bags : ℕ := 2

/-- The total amount spent on the bags after discount -/
def total_spent : ℝ := 3

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  (1 - discount_percentage) * (num_bags * original_price) = total_spent := by
  sorry

end original_price_correct_l2313_231331


namespace pen_and_notebook_cost_l2313_231393

theorem pen_and_notebook_cost (pen_cost : ℝ) (price_difference : ℝ) : 
  pen_cost = 4.5 → price_difference = 1.8 → pen_cost + (pen_cost - price_difference) = 7.2 := by
  sorry

end pen_and_notebook_cost_l2313_231393


namespace simplest_quadratic_radical_l2313_231399

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ) (b : ℕ), x = a * Real.sqrt b ∧ 
  (∀ (c : ℚ) (d : ℕ), x = c * Real.sqrt d → b ≤ d)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (-Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/7)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) :=
by sorry

end simplest_quadratic_radical_l2313_231399


namespace boxes_per_hand_for_seven_people_l2313_231398

/-- Given a group of people and the total number of boxes they can hold, 
    calculate the number of boxes one person can hold in each hand. -/
def boxes_per_hand (num_people : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / num_people) / 2

/-- Theorem stating that given 7 people holding 14 boxes in total, 
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_seven_people : 
  boxes_per_hand 7 14 = 1 := by sorry

end boxes_per_hand_for_seven_people_l2313_231398


namespace expression_simplification_l2313_231301

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end expression_simplification_l2313_231301


namespace equation_solution_l2313_231316

theorem equation_solution : ∃! x : ℝ, (x - 1) + 2 * Real.sqrt (x + 3) = 5 := by sorry

end equation_solution_l2313_231316


namespace charles_whistles_l2313_231397

/-- Given that Sean has 45 whistles and 32 more whistles than Charles,
    prove that Charles has 13 whistles. -/
theorem charles_whistles (sean_whistles : ℕ) (difference : ℕ) :
  sean_whistles = 45 →
  difference = 32 →
  sean_whistles - difference = 13 :=
by
  sorry

end charles_whistles_l2313_231397


namespace two_numbers_with_given_means_l2313_231392

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5)) := by
sorry

end two_numbers_with_given_means_l2313_231392


namespace lcm_36_100_l2313_231335

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end lcm_36_100_l2313_231335


namespace ax5_plus_by5_l2313_231306

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^5 + b * y^5 = 6200 / 29 := by
sorry

end ax5_plus_by5_l2313_231306


namespace odd_function_iff_condition_l2313_231354

def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_iff_condition (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by sorry

end odd_function_iff_condition_l2313_231354


namespace geometric_sequence_common_ratio_l2313_231302

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_neg_first : a 1 < 0)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  ∃ q : ℝ, 0 < q ∧ q < 1 :=
sorry

end geometric_sequence_common_ratio_l2313_231302


namespace minimum_additional_stickers_l2313_231377

def initial_stickers : ℕ := 29
def row_size : ℕ := 4
def group_size : ℕ := 5

theorem minimum_additional_stickers :
  let total_stickers := initial_stickers + 11
  (total_stickers % row_size = 0) ∧
  (total_stickers % group_size = 0) ∧
  (∀ n : ℕ, n < 11 →
    let test_total := initial_stickers + n
    (test_total % row_size ≠ 0) ∨ (test_total % group_size ≠ 0)) :=
by sorry

end minimum_additional_stickers_l2313_231377


namespace problem_solution_l2313_231315

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (A = {x : ℝ | -3 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 5}) ∧
  (∀ m : ℝ, A ∪ B m = A ↔ m ≤ 5/2) := by
  sorry

end problem_solution_l2313_231315


namespace negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l2313_231328

theorem negation_of_forall_gt (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_is_le {a b : ℝ} :
  ¬(a > b) ↔ (a ≤ b) :=
by sorry

theorem negation_of_forall_x_squared_gt_1_minus_2x :
  (¬∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) :=
by sorry

end negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l2313_231328


namespace shortening_theorem_l2313_231357

/-- A sequence of digits where each digit is 0 or 9 -/
def DigitSequence := List (Fin 2)

/-- The length of the original sequence -/
def originalLength : Nat := 2015

/-- The probability of a digit being the same as the previous one -/
def sameDigitProb : Real := 0.1

/-- The probability of a digit being different from the previous one -/
def differentDigitProb : Real := 0.9

/-- The shortening operation on a digit sequence -/
def shortenSequence (seq : DigitSequence) : DigitSequence :=
  sorry

/-- The probability that the sequence will shorten by exactly one digit -/
def shortenByOneProb (n : Nat) : Real :=
  sorry

/-- The expected length of the new sequence after shortening -/
def expectedNewLength (n : Nat) : Real :=
  sorry

theorem shortening_theorem :
  ∃ (ε : Real),
    ε > 0 ∧
    ε < 1e-89 ∧
    abs (shortenByOneProb originalLength - 1.564e-90) < ε ∧
    abs (expectedNewLength originalLength - 1813.6) < ε :=
  sorry

end shortening_theorem_l2313_231357


namespace pizza_toppings_combinations_l2313_231341

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + n.choose 2 + n.choose 3 = 92 := by
  sorry

end pizza_toppings_combinations_l2313_231341


namespace max_team_size_l2313_231311

/-- A function that represents a valid selection of team numbers -/
def ValidSelection (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, x ≤ 100 ∧
  ∀ y ∈ s, ∀ z ∈ s, x ≠ y + z ∧
  ∀ y ∈ s, x ≠ 2 * y

/-- The theorem stating the maximum size of a valid selection is 50 -/
theorem max_team_size :
  (∃ s : Finset ℕ, ValidSelection s ∧ s.card = 50) ∧
  ∀ s : Finset ℕ, ValidSelection s → s.card ≤ 50 := by sorry

end max_team_size_l2313_231311


namespace units_digit_of_quotient_l2313_231385

theorem units_digit_of_quotient (n : ℕ) : (2^2023 + 3^2023) % 7 = 0 → 
  (2^2023 + 3^2023) / 7 % 10 = 0 := by
  sorry

end units_digit_of_quotient_l2313_231385


namespace max_x_minus_y_l2313_231359

-- Define the condition function
def condition (x y : ℝ) : Prop := 3 * (x^2 + y^2) = x^2 + y

-- Define the objective function
def objective (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem max_x_minus_y :
  ∃ (max : ℝ), max = 1 / Real.sqrt 24 ∧
  ∀ (x y : ℝ), condition x y → objective x y ≤ max :=
sorry

end max_x_minus_y_l2313_231359


namespace encoded_equation_unique_solution_l2313_231329

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- The encoded equation -/
def EncodedEquation (Δ square triangle circle : Digit) : Prop :=
  ∃ (base : TwoDigitNumber) (result : ThreeDigitNumber),
    base.val = 10 * Δ.val + square.val ∧
    result.val = 100 * square.val + 10 * circle.val + square.val ∧
    base.val ^ triangle.val = result.val

theorem encoded_equation_unique_solution :
  ∃! (Δ square triangle circle : Digit), EncodedEquation Δ square triangle circle :=
sorry

end encoded_equation_unique_solution_l2313_231329


namespace triangle_max_area_l2313_231381

/-- Given a triangle ABC with side a = √2 and acosB + bsinA = c, 
    the maximum area of the triangle is (√2 + 1) / 2 -/
theorem triangle_max_area (a b c A B C : Real) :
  a = Real.sqrt 2 →
  a * Real.cos B + b * Real.sin A = c →
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
    S ≤ (Real.sqrt 2 + 1) / 2 ∧
    (S = (Real.sqrt 2 + 1) / 2 ↔ b = c)) :=
by sorry

end triangle_max_area_l2313_231381


namespace geometric_sequence_sum_l2313_231374

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l2313_231374


namespace max_slope_no_lattice_points_l2313_231339

theorem max_slope_no_lattice_points :
  let max_a : ℚ := 25 / 49
  ∀ a : ℚ, (∀ m x y : ℚ,
    (1 / 2 < m) → (m < a) →
    (0 < x) → (x ≤ 50) →
    (y = m * x + 3) →
    (∃ n : ℤ, x = ↑n) →
    (∃ n : ℤ, y = ↑n) →
    False) →
  a ≤ max_a :=
by sorry

end max_slope_no_lattice_points_l2313_231339


namespace derivative_at_negative_one_l2313_231347

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_at_negative_one (a b c : ℝ) :
  let f := fun x : ℝ => a * x^4 + b * x^2 + c
  (deriv f) 1 = 2 → (deriv f) (-1) = -2 := by
sorry

end derivative_at_negative_one_l2313_231347


namespace quadratic_expression_value_l2313_231358

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
sorry

end quadratic_expression_value_l2313_231358


namespace square_division_theorem_l2313_231326

/-- Represents a cell in the square array -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a rectangle in the square array -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- Represents the state of a cell (pink or not pink) -/
inductive CellState
  | Pink
  | NotPink

/-- Represents the square array -/
def SquareArray (n : Nat) := Fin n → Fin n → CellState

/-- Checks if a rectangle contains exactly one pink cell -/
def containsOnePinkCell (arr : SquareArray n) (rect : Rectangle) : Prop := sorry

/-- Checks if a list of rectangles forms a valid division of the square -/
def isValidDivision (n : Nat) (rectangles : List Rectangle) : Prop := sorry

/-- The main theorem -/
theorem square_division_theorem (n : Nat) (arr : SquareArray n) :
  (∃ (i j : Fin n), arr i j = CellState.Pink) →
  ∃ (rectangles : List Rectangle), 
    isValidDivision n rectangles ∧ 
    ∀ rect ∈ rectangles, containsOnePinkCell arr rect :=
sorry

end square_division_theorem_l2313_231326


namespace rectangle_area_doubling_l2313_231323

theorem rectangle_area_doubling (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  let new_length := 1.4 * l
  let new_width := (10/7) * w
  let original_area := l * w
  let new_area := new_length * new_width
  new_area = 2 * original_area := by sorry

end rectangle_area_doubling_l2313_231323


namespace pet_ownership_l2313_231325

theorem pet_ownership (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 28)
  (h3 : cat_owners = 35)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 13 := by
sorry

end pet_ownership_l2313_231325


namespace max_red_socks_l2313_231372

theorem max_red_socks (x y : ℕ) : 
  x + y ≤ 2017 →
  (x * (x - 1) + y * (y - 1)) / ((x + y) * (x + y - 1)) = 1 / 2 →
  x ≤ 990 :=
by sorry

end max_red_socks_l2313_231372


namespace correct_sunset_time_l2313_231353

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + (t.hours + d.hours) * 60
  { hours := totalMinutes / 60 % 24,
    minutes := totalMinutes % 60 }

def sunsetTime (sunrise : Time) (daylight : Duration) : Time :=
  addTime sunrise daylight

theorem correct_sunset_time :
  let sunrise : Time := { hours := 16, minutes := 35 }
  let daylight : Duration := { hours := 9, minutes := 48 }
  sunsetTime sunrise daylight = { hours := 2, minutes := 23 } := by
  sorry

end correct_sunset_time_l2313_231353


namespace remainder_sum_l2313_231305

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 74) 
  (hb : b % 120 = 114) : 
  (a + b) % 40 = 28 := by
  sorry

end remainder_sum_l2313_231305


namespace hemisphere_chord_length_l2313_231324

theorem hemisphere_chord_length (R : ℝ) (h : R = 20) : 
  let chord_length := 2 * R * Real.sqrt 2 / 2
  chord_length = 20 * Real.sqrt 2 := by
  sorry

#check hemisphere_chord_length

end hemisphere_chord_length_l2313_231324


namespace open_box_volume_is_24000_l2313_231340

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a parallelogram cut from corners -/
structure ParallelogramCut where
  base : ℝ
  height : ℝ

/-- Calculates the volume of the open box created from a sheet with given dimensions and corner cuts -/
def openBoxVolume (sheet : SheetDimensions) (cut : ParallelogramCut) : ℝ :=
  (sheet.length - 2 * cut.base) * (sheet.width - 2 * cut.base) * cut.height

/-- Theorem stating that the volume of the open box is 24000 m^3 -/
theorem open_box_volume_is_24000 (sheet : SheetDimensions) (cut : ParallelogramCut)
    (h1 : sheet.length = 100)
    (h2 : sheet.width = 50)
    (h3 : cut.base = 10)
    (h4 : cut.height = 10) :
    openBoxVolume sheet cut = 24000 := by
  sorry

end open_box_volume_is_24000_l2313_231340


namespace sons_age_l2313_231389

theorem sons_age (son_age man_age : ℕ) : 
  (man_age = son_age + 20) →
  (man_age + 2 = 2 * (son_age + 2)) →
  son_age = 18 := by
sorry

end sons_age_l2313_231389


namespace sticks_per_hour_to_stay_warm_l2313_231368

/-- The number of sticks of wood produced by chopping up furniture -/
def sticks_per_furniture : Nat → Nat
| 0 => 6  -- chairs
| 1 => 9  -- tables
| 2 => 2  -- stools
| _ => 0  -- other furniture (not considered)

/-- The number of each type of furniture Mary chopped up -/
def furniture_count : Nat → Nat
| 0 => 18  -- chairs
| 1 => 6   -- tables
| 2 => 4   -- stools
| _ => 0   -- other furniture (not considered)

/-- The number of hours Mary can keep warm -/
def warm_hours : Nat := 34

/-- Calculates the total number of sticks of wood Mary has -/
def total_sticks : Nat :=
  (sticks_per_furniture 0 * furniture_count 0) +
  (sticks_per_furniture 1 * furniture_count 1) +
  (sticks_per_furniture 2 * furniture_count 2)

/-- The theorem to prove -/
theorem sticks_per_hour_to_stay_warm :
  total_sticks / warm_hours = 5 := by
  sorry

end sticks_per_hour_to_stay_warm_l2313_231368


namespace rhombus_longer_diagonal_l2313_231383

theorem rhombus_longer_diagonal (side_length shorter_diagonal : ℝ) :
  side_length = 65 ∧ shorter_diagonal = 72 →
  ∃ longer_diagonal : ℝ, longer_diagonal = 108 ∧
  longer_diagonal^2 + shorter_diagonal^2 = 4 * side_length^2 :=
by sorry

end rhombus_longer_diagonal_l2313_231383


namespace integer_less_than_sqrt_23_l2313_231388

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end integer_less_than_sqrt_23_l2313_231388


namespace angle_half_in_second_quadrant_l2313_231380

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

def is_in_second_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi + Real.pi / 2 < θ ∧ θ < k * Real.pi + Real.pi

theorem angle_half_in_second_quadrant (α : Real) 
  (h1 : is_in_third_quadrant α) 
  (h2 : |Real.cos (α/2)| = -Real.cos (α/2)) : 
  is_in_second_quadrant (α/2) := by
  sorry


end angle_half_in_second_quadrant_l2313_231380


namespace calculation_proofs_l2313_231387

theorem calculation_proofs :
  (40 + (1/6 - 2/3 + 3/4) * 12 = 43) ∧
  ((-1)^2021 + |(-9)| * (2/3) + (-3) / (1/5) = -10) := by
  sorry

end calculation_proofs_l2313_231387


namespace fifth_term_ratio_l2313_231327

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively -/
def arithmetic_sequences (a b : ℕ → ℝ) (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of sums S_n and T_n is 2n / (3n + 1) -/
def sum_ratio (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n / T n = (2 * n : ℝ) / (3 * n + 1)

/-- The main theorem: given the conditions, prove a_5 / b_5 = 9 / 14 -/
theorem fifth_term_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ)
    (h1 : arithmetic_sequences a b S T) (h2 : sum_ratio S T) :
    a 5 / b 5 = 9 / 14 := by
  sorry

end fifth_term_ratio_l2313_231327


namespace factorization_equality_l2313_231355

theorem factorization_equality (a b : ℝ) : 3 * a * b^2 + a^2 * b = a * b * (3 * b + a) := by
  sorry

end factorization_equality_l2313_231355


namespace sixth_power_of_complex_root_of_unity_l2313_231320

theorem sixth_power_of_complex_root_of_unity (z : ℂ) : 
  z = (-1 + Complex.I * Real.sqrt 3) / 2 → z^6 = (1 : ℂ) / 4 := by
  sorry

end sixth_power_of_complex_root_of_unity_l2313_231320


namespace circular_lake_diameter_l2313_231312

/-- The diameter of a circular lake with radius 7 meters is 14 meters. -/
theorem circular_lake_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end circular_lake_diameter_l2313_231312


namespace type_of_2004_least_type_B_after_2004_l2313_231308

/-- Represents the type of a number in the game -/
inductive NumberType
| A
| B

/-- Determines if a number is of type A or B in the game -/
def numberType (n : ℕ) : NumberType :=
  sorry

/-- Theorem stating that 2004 is of type A -/
theorem type_of_2004 : numberType 2004 = NumberType.A :=
  sorry

/-- Theorem stating that 2048 is the least number greater than 2004 of type B -/
theorem least_type_B_after_2004 :
  (numberType 2048 = NumberType.B) ∧
  (∀ m : ℕ, 2004 < m → m < 2048 → numberType m = NumberType.A) :=
  sorry

end type_of_2004_least_type_B_after_2004_l2313_231308


namespace smallest_odd_with_24_divisors_l2313_231373

/-- The number of divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_odd_with_24_divisors :
  ∃ (n : ℕ+),
    isOdd n.val ∧
    numDivisors n = 24 ∧
    (∀ (m : ℕ+), isOdd m.val ∧ numDivisors m = 24 → n ≤ m) ∧
    n = 3465 := by sorry

end smallest_odd_with_24_divisors_l2313_231373


namespace dancer_count_l2313_231352

theorem dancer_count (n : ℕ) : 
  (200 ≤ n ∧ n ≤ 300) ∧
  (∃ k : ℕ, n + 5 = 12 * k) ∧
  (∃ m : ℕ, n + 5 = 10 * m) →
  n = 235 ∨ n = 295 := by
sorry

end dancer_count_l2313_231352


namespace diophantine_equation_solutions_l2313_231350

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by sorry

end diophantine_equation_solutions_l2313_231350


namespace number_puzzle_l2313_231332

theorem number_puzzle (x : ℝ) : (x / 2) / 2 = 85 + 45 → x - 45 = 475 := by
  sorry

end number_puzzle_l2313_231332


namespace box_dimensions_sum_l2313_231366

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Defines the properties of the rectangular box -/
def validBox (d : BoxDimensions) : Prop :=
  d.A * d.B = 18 ∧ d.A * d.C = 32 ∧ d.B * d.C = 50

/-- Theorem stating that the sum of dimensions is approximately 57.28 -/
theorem box_dimensions_sum (d : BoxDimensions) (h : validBox d) :
  ∃ ε > 0, |d.A + d.B + d.C - 57.28| < ε :=
sorry

end box_dimensions_sum_l2313_231366


namespace workshop_workers_workshop_workers_proof_l2313_231317

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers : ℕ :=
  let avg_salary : ℚ := 8000
  let num_technicians : ℕ := 7
  let avg_salary_technicians : ℚ := 18000
  let avg_salary_others : ℚ := 6000
  42

/-- Proof that the total number of workers is 42 -/
theorem workshop_workers_proof : workshop_workers = 42 := by
  sorry

end workshop_workers_workshop_workers_proof_l2313_231317


namespace weight_lifting_multiple_l2313_231367

theorem weight_lifting_multiple (rodney roger ron : ℕ) (m : ℕ) : 
  rodney + roger + ron = 239 →
  rodney = 2 * roger →
  roger = m * ron - 7 →
  rodney = 146 →
  m = 4 := by
sorry

end weight_lifting_multiple_l2313_231367


namespace integer_roots_of_cubic_l2313_231390

def p (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

theorem integer_roots_of_cubic :
  {x : ℤ | p x = 0} = {-1, -2, 3} := by sorry

end integer_roots_of_cubic_l2313_231390


namespace draw_balls_count_l2313_231349

/-- The number of ways to draw 4 balls from 20 balls numbered 1 through 20,
    where the sum of the first and last ball drawn is 21. -/
def draw_balls : ℕ :=
  let total_balls : ℕ := 20
  let balls_drawn : ℕ := 4
  let sum_first_last : ℕ := 21
  let valid_first_balls : ℕ := sum_first_last - 1
  let remaining_choices : ℕ := total_balls - 2
  valid_first_balls * remaining_choices * (remaining_choices - 1)

theorem draw_balls_count : draw_balls = 3060 := by
  sorry

end draw_balls_count_l2313_231349


namespace tangent_slope_at_point_l2313_231348

-- Define the function representing the curve
def f (x : ℝ) := x^2 + 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point : 
  f 2 = 10 → f' 2 = 7 :=
by
  sorry

end tangent_slope_at_point_l2313_231348


namespace milk_remaining_l2313_231364

theorem milk_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 8 → given = 18/7 → remaining = initial - given → remaining = 38/7 := by
  sorry

end milk_remaining_l2313_231364


namespace triangle_legs_theorem_l2313_231370

/-- A point inside a right angle -/
structure PointInRightAngle where
  /-- Distance from the point to one side of the angle -/
  dist1 : ℝ
  /-- Distance from the point to the other side of the angle -/
  dist2 : ℝ

/-- A triangle formed by a line through a point in a right angle -/
structure TriangleInRightAngle where
  /-- The point inside the right angle -/
  point : PointInRightAngle
  /-- The area of the triangle -/
  area : ℝ

/-- The legs of a right triangle -/
structure RightTriangleLegs where
  /-- Length of one leg -/
  leg1 : ℝ
  /-- Length of the other leg -/
  leg2 : ℝ

/-- Theorem about the legs of a specific triangle in a right angle -/
theorem triangle_legs_theorem (t : TriangleInRightAngle)
    (h1 : t.point.dist1 = 4)
    (h2 : t.point.dist2 = 8)
    (h3 : t.area = 100) :
    (∃ l : RightTriangleLegs, (l.leg1 = 40 ∧ l.leg2 = 5) ∨ (l.leg1 = 10 ∧ l.leg2 = 20)) :=
  sorry

end triangle_legs_theorem_l2313_231370


namespace area_at_stage_8_l2313_231378

/-- Calculates the number of squares added up to a given stage -/
def squaresAdded (stage : ℕ) : ℕ :=
  (stage + 1) / 2

/-- The side length of each square in inches -/
def squareSideLength : ℕ := 4

/-- Calculates the area of the figure at a given stage -/
def areaAtStage (stage : ℕ) : ℕ :=
  (squaresAdded stage) * (squareSideLength * squareSideLength)

/-- Proves that the area of the figure at Stage 8 is 64 square inches -/
theorem area_at_stage_8 : areaAtStage 8 = 64 := by
  sorry

end area_at_stage_8_l2313_231378


namespace beth_crayons_l2313_231337

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 8

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 12

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 15

/-- The number of crayons Beth borrowed from her friend -/
def borrowed_crayons : ℕ := 7

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons + borrowed_crayons

theorem beth_crayons :
  total_crayons = 118 := by
  sorry

end beth_crayons_l2313_231337


namespace first_half_speed_l2313_231371

/-- Proves that given a journey of 3600 miles completed in 30 hours, 
    where the second half is traveled at 180 mph, 
    the average speed for the first half of the journey is 90 mph. -/
theorem first_half_speed (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 3600 →
  total_time = 30 →
  second_half_speed = 180 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 90 :=
by sorry

end first_half_speed_l2313_231371


namespace equation_solutions_l2313_231330

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end equation_solutions_l2313_231330


namespace segment_bisection_l2313_231379

-- Define the angle
structure Angle where
  C : Point
  K : Point
  L : Point

-- Define the condition for a point to be inside an angle
def InsideAngle (α : Angle) (O : Point) : Prop := sorry

-- Define the condition for a point to be on a line
def OnLine (P Q R : Point) : Prop := sorry

-- Define the midpoint of a segment
def Midpoint (M A B : Point) : Prop := sorry

-- Main theorem
theorem segment_bisection (α : Angle) (O : Point) 
  (h : InsideAngle α O) : 
  ∃ (A B : Point), 
    OnLine α.C α.K A ∧ 
    OnLine α.C α.L B ∧ 
    Midpoint O A B :=
  sorry

end segment_bisection_l2313_231379


namespace max_player_salary_l2313_231343

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  num_players = 25 →
  min_salary = 18000 →
  total_cap = 900000 →
  (num_players - 1) * min_salary + (total_cap - (num_players - 1) * min_salary) ≤ total_cap →
  (∀ (salaries : List ℕ), salaries.length = num_players → 
    (∀ s ∈ salaries, s ≥ min_salary) → 
    salaries.sum ≤ total_cap →
    ∀ s ∈ salaries, s ≤ 468000) :=
by sorry

#check max_player_salary

end max_player_salary_l2313_231343


namespace mass_of_man_in_boat_l2313_231361

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth : Real) : Real :=
  boat_length * boat_breadth * sink_depth * 1000

/-- Theorem stating the mass of the man in the given problem. -/
theorem mass_of_man_in_boat : mass_of_man 3 2 0.02 = 120 := by
  sorry

#eval mass_of_man 3 2 0.02

end mass_of_man_in_boat_l2313_231361


namespace zoe_pool_cleaning_earnings_l2313_231395

/-- Represents Zoe's earnings and babysitting information --/
structure ZoeEarnings where
  total : ℕ
  zacharyRate : ℕ
  julieRate : ℕ
  chloeRate : ℕ
  zacharyEarnings : ℕ

/-- Calculates Zoe's earnings from pool cleaning --/
def poolCleaningEarnings (z : ZoeEarnings) : ℕ :=
  let zacharyHours := z.zacharyEarnings / z.zacharyRate
  let chloeHours := zacharyHours * 5
  let julieHours := zacharyHours * 3
  let babysittingEarnings := 
    zacharyHours * z.zacharyRate + 
    chloeHours * z.chloeRate + 
    julieHours * z.julieRate
  z.total - babysittingEarnings

/-- Theorem stating that Zoe's pool cleaning earnings are $5,200 --/
theorem zoe_pool_cleaning_earnings :
  poolCleaningEarnings {
    total := 8000,
    zacharyRate := 15,
    julieRate := 10,
    chloeRate := 5,
    zacharyEarnings := 600
  } = 5200 := by
  sorry

end zoe_pool_cleaning_earnings_l2313_231395


namespace smallest_k_no_real_roots_l2313_231382

theorem smallest_k_no_real_roots :
  ∀ k : ℤ, (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) → k ≥ 4 :=
by sorry

end smallest_k_no_real_roots_l2313_231382


namespace multiples_of_seven_l2313_231300

theorem multiples_of_seven (a b : ℤ) (q : Set ℤ) : 
  (∃ k₁ k₂ : ℤ, a = 14 * k₁ ∧ b = 14 * k₂) →
  q = {x : ℤ | a ≤ x ∧ x ≤ b} →
  (Finset.filter (fun x => x % 14 = 0) (Finset.Icc a b)).card = 12 →
  (Finset.filter (fun x => x % 7 = 0) (Finset.Icc a b)).card = 24 := by
sorry

end multiples_of_seven_l2313_231300


namespace all_events_probability_at_least_one_not_occurring_l2313_231345

-- Define the probabilities of each event
def P_A : ℝ := 0.8
def P_B : ℝ := 0.6
def P_C : ℝ := 0.5

-- Theorem for the probability of all three events occurring
theorem all_events_probability :
  P_A * P_B * P_C = 0.24 :=
sorry

-- Theorem for the probability of at least one event not occurring
theorem at_least_one_not_occurring :
  1 - (P_A * P_B * P_C) = 0.76 :=
sorry

end all_events_probability_at_least_one_not_occurring_l2313_231345


namespace trig_identity_l2313_231322

theorem trig_identity (x y : ℝ) :
  (Real.sin x)^2 + (Real.sin (x + y + π/4))^2 - 
  2 * (Real.sin x) * (Real.sin (y + π/4)) * (Real.sin (x + y + π/4)) = 
  1 - (1/2) * (Real.sin y)^2 := by
sorry

end trig_identity_l2313_231322


namespace melanie_dimes_l2313_231304

/-- The number of dimes Melanie initially had -/
def initial_dimes : ℕ := 7

/-- The number of dimes Melanie gave to her dad -/
def dimes_to_dad : ℕ := 8

/-- The number of dimes Melanie's mother gave her -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 3

theorem melanie_dimes : 
  initial_dimes - dimes_to_dad + dimes_from_mother = current_dimes :=
by sorry

end melanie_dimes_l2313_231304


namespace square_diagonal_less_than_twice_fg_l2313_231303

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B = (A.1 + s, A.2) ∧
    C = (A.1 + s, A.2 + s) ∧
    D = (A.1, A.2 + s)

-- Define that E is an internal point on side AD
def InternalPointOnSide (E A D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (A.1, A.2 + t * (D.2 - A.2))

-- Define F as the foot of the perpendicular from B to CE
def PerpendicularFoot (F B C E : ℝ × ℝ) : Prop :=
  (F.1 - C.1) * (E.1 - C.1) + (F.2 - C.2) * (E.2 - C.2) = 0 ∧
  (F.1 - B.1) * (E.1 - C.1) + (F.2 - B.2) * (E.2 - C.2) = 0

-- Define that BG = FG
def EqualDistances (B F G : ℝ × ℝ) : Prop :=
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = (G.1 - F.1)^2 + (G.2 - F.2)^2

-- Define that the line through G parallel to BC passes through the midpoint of EF
def ParallelThroughMidpoint (G B C E F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = ((E.1 + F.1)/2 + t*(C.1 - B.1), (E.2 + F.2)/2 + t*(C.2 - B.2))

-- State the theorem
theorem square_diagonal_less_than_twice_fg 
  (A B C D E F G : ℝ × ℝ) : 
  Square A B C D → 
  InternalPointOnSide E A D → 
  PerpendicularFoot F B C E → 
  EqualDistances B F G → 
  ParallelThroughMidpoint G B C E F → 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 < 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
by sorry

end square_diagonal_less_than_twice_fg_l2313_231303
