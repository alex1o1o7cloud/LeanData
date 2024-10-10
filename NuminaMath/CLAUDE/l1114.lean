import Mathlib

namespace ratio_a_to_c_l1114_111490

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 3 / 10) :
  a / c = 25 / 12 := by
  sorry

end ratio_a_to_c_l1114_111490


namespace solution_range_l1114_111459

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (2 * x - 1) / (x + 1) = 3 - m / (x + 1)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ equation x m ∧ x ≠ -1) → m < 4 ∧ m ≠ 3 :=
by sorry

end solution_range_l1114_111459


namespace marbles_distribution_l1114_111404

/-- Given 20 marbles distributed equally among 2 boys, prove that each boy receives 10 marbles. -/
theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 20 →
  num_boys = 2 →
  marbles_per_boy * num_boys = total_marbles →
  marbles_per_boy = 10 := by
  sorry

end marbles_distribution_l1114_111404


namespace photo_arrangements_l1114_111446

/-- Represents the number of students in the photo --/
def num_students : ℕ := 5

/-- Represents the constraint that B and C must stand together --/
def bc_together : Prop := True

/-- Represents the constraint that A cannot stand next to B --/
def a_not_next_to_b : Prop := True

/-- The number of different arrangements --/
def num_arrangements : ℕ := 36

/-- Theorem stating that the number of arrangements is 36 --/
theorem photo_arrangements :
  (num_students = 5) →
  bc_together →
  a_not_next_to_b →
  num_arrangements = 36 := by
  sorry

end photo_arrangements_l1114_111446


namespace cake_recipe_flour_calculation_l1114_111483

/-- Given a ratio of milk to flour and an amount of milk used, calculate the amount of flour needed. -/
def flour_needed (milk_ratio : ℚ) (flour_ratio : ℚ) (milk_used : ℚ) : ℚ :=
  (flour_ratio / milk_ratio) * milk_used

/-- The theorem states that given the specified ratio and milk amount, the flour needed is 1200 mL. -/
theorem cake_recipe_flour_calculation :
  let milk_ratio : ℚ := 60
  let flour_ratio : ℚ := 300
  let milk_used : ℚ := 240
  flour_needed milk_ratio flour_ratio milk_used = 1200 := by
sorry

#eval flour_needed 60 300 240

end cake_recipe_flour_calculation_l1114_111483


namespace unique_number_with_gcd_l1114_111449

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end unique_number_with_gcd_l1114_111449


namespace combined_salaries_l1114_111466

/-- The combined salaries of B, C, D, and E given A's salary and the average salary of all five -/
theorem combined_salaries 
  (salary_A : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_A = 8000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_A = 35000 := by
  sorry

end combined_salaries_l1114_111466


namespace smallest_integer_satisfying_inequality_l1114_111499

theorem smallest_integer_satisfying_inequality : 
  ∀ y : ℤ, y < 3 * y - 14 → y ≥ 8 ∧ 8 < 3 * 8 - 14 := by
  sorry

end smallest_integer_satisfying_inequality_l1114_111499


namespace nested_fraction_equality_l1114_111424

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end nested_fraction_equality_l1114_111424


namespace project_hours_difference_l1114_111439

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 189) 
  (kate_hours : ℕ) (pat_hours : ℕ) (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours = mark_hours / 3) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 105 := by
sorry

end project_hours_difference_l1114_111439


namespace probability_two_boys_or_two_girls_l1114_111469

/-- The probability of selecting either two boys or two girls from a group of 5 students -/
theorem probability_two_boys_or_two_girls (total_students : ℕ) (num_boys : ℕ) (num_girls : ℕ) :
  total_students = 5 →
  num_boys = 2 →
  num_girls = 3 →
  (Nat.choose num_girls 2 + Nat.choose num_boys 2) / Nat.choose total_students 2 = 2 / 5 := by
  sorry

end probability_two_boys_or_two_girls_l1114_111469


namespace smallest_sum_of_three_positive_l1114_111495

def S : Set Int := {2, 5, -7, 8, -10}

theorem smallest_sum_of_three_positive : 
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b + c = 15 ∧
   (∀ (x y z : Int), x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y → y ≠ z → x ≠ z → 
    x > 0 → y > 0 → z > 0 → 
    x + y + z ≥ 15)) := by
  sorry

#check smallest_sum_of_three_positive

end smallest_sum_of_three_positive_l1114_111495


namespace velocity_at_t_1_is_zero_l1114_111432

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2*t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := -2*t + 2

-- Theorem statement
theorem velocity_at_t_1_is_zero : v 1 = 0 := by
  sorry

end velocity_at_t_1_is_zero_l1114_111432


namespace mall_entrance_exit_ways_l1114_111457

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

#check mall_entrance_exit_ways

end mall_entrance_exit_ways_l1114_111457


namespace count_integers_eq_25_l1114_111436

/-- The number of integers between 100 and 200 (exclusive) that have the same remainder when divided by 6 and 8 -/
def count_integers : ℕ :=
  (Finset.filter (λ n : ℕ => 
    100 < n ∧ n < 200 ∧ n % 6 = n % 8
  ) (Finset.range 200)).card

/-- Theorem stating that there are exactly 25 such integers -/
theorem count_integers_eq_25 : count_integers = 25 := by
  sorry

end count_integers_eq_25_l1114_111436


namespace tara_yoghurt_purchase_l1114_111453

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := sorry

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := 19

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- The difference in dollars between ice cream and yoghurt spending -/
def spending_difference : ℕ := 129

theorem tara_yoghurt_purchase : 
  ice_cream_cartons * ice_cream_cost = 
  yoghurt_cartons * yoghurt_cost + spending_difference ∧ 
  yoghurt_cartons = 4 := by sorry

end tara_yoghurt_purchase_l1114_111453


namespace intersection_of_A_and_B_l1114_111420

def set_A : Set ℝ := {x | |x| ≤ 1}
def set_B : Set ℝ := {y | ∃ x, y = x^2}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l1114_111420


namespace triangle_area_from_rectangle_ratio_l1114_111405

/-- Given a rectangle with length 6 cm and width 4 cm, and a triangle whose area is in a 5:2 ratio
    with the rectangle's area, prove that the area of the triangle is 60 cm². -/
theorem triangle_area_from_rectangle_ratio :
  ∀ (rectangle_length rectangle_width triangle_area : ℝ),
  rectangle_length = 6 →
  rectangle_width = 4 →
  5 * (rectangle_length * rectangle_width) = 2 * triangle_area →
  triangle_area = 60 :=
by
  sorry

end triangle_area_from_rectangle_ratio_l1114_111405


namespace sum_of_abc_l1114_111476

theorem sum_of_abc (a b c : ℝ) : 
  a * (a - 4) = 5 →
  b * (b - 4) = 5 →
  c * (c - 4) = 5 →
  a^2 + b^2 = c^2 →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  a + b + c = 4 + Real.sqrt 26 := by
  sorry

end sum_of_abc_l1114_111476


namespace propositions_truth_l1114_111473

-- Define the necessary geometric objects
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relations
def subset (a : Line) (α : Plane) : Prop := sorry
def perpendicular_line_plane (a : Line) (β : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- Define the propositions
def proposition_p (a : Line) (α β : Plane) : Prop :=
  subset a α → (perpendicular_line_plane a β → perpendicular_planes α β)

-- Define a polyhedron type
def Polyhedron : Type := sorry

-- Define the properties of a polyhedron
def has_two_parallel_faces (p : Polyhedron) : Prop := sorry
def other_faces_are_trapezoids (p : Polyhedron) : Prop := sorry
def is_prism (p : Polyhedron) : Prop := sorry

-- Define proposition q
def proposition_q (p : Polyhedron) : Prop :=
  has_two_parallel_faces p ∧ other_faces_are_trapezoids p → is_prism p

theorem propositions_truth : ∃ (a : Line) (α β : Plane) (p : Polyhedron),
  proposition_p a α β ∧ ¬proposition_q p := by
  sorry

end propositions_truth_l1114_111473


namespace fixed_point_of_square_minus_600_l1114_111462

theorem fixed_point_of_square_minus_600 :
  ∃! (x : ℕ), x = x^2 - 600 :=
by
  -- The unique natural number satisfying the equation is 25
  use 25
  constructor
  · -- Prove that 25 satisfies the equation
    norm_num
  · -- Prove that any natural number satisfying the equation must be 25
    intro y hy
    -- Here we would prove that y = 25
    sorry

#eval (25 : ℕ)^2 - 600  -- This should evaluate to 25

end fixed_point_of_square_minus_600_l1114_111462


namespace q_satisfies_conditions_l1114_111440

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (16/7) * x^2 + (32/7) * x - 240/7

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions : 
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -16 := by
  sorry

end q_satisfies_conditions_l1114_111440


namespace quadratic_expression_value_l1114_111430

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 12) 
  (eq2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := by sorry

end quadratic_expression_value_l1114_111430


namespace geometric_sequence_sum_l1114_111421

/-- Given an infinite geometric sequence {a_n} with first term 1 and common ratio a - 3/2,
    if the sum of all terms is a, then a = 2. -/
theorem geometric_sequence_sum (a : ℝ) : 
  let a_1 : ℝ := 1
  let q : ℝ := a - 3/2
  let sum : ℝ := a_1 / (1 - q)
  (sum = a) → (a = 2) := by
  sorry

end geometric_sequence_sum_l1114_111421


namespace floor_equation_solution_l1114_111433

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3*x⌋ + 1/3⌋ = ⌊x + 3⌋) ↔ (4/3 ≤ x ∧ x < 5/3) := by
  sorry

end floor_equation_solution_l1114_111433


namespace range_of_fraction_l1114_111450

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ x, -2 < x ∧ x < -1/2 ∧ x = a/b :=
by sorry

end range_of_fraction_l1114_111450


namespace negative_irrational_less_than_neg_three_l1114_111428

theorem negative_irrational_less_than_neg_three :
  ∃ x : ℝ, x < -3 ∧ Irrational x ∧ x < 0 :=
by
  -- Proof goes here
  sorry

end negative_irrational_less_than_neg_three_l1114_111428


namespace quadratic_rewrite_l1114_111426

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, (d * x + e)^2 + f = 4 * x^2 - 28 * x + 49) →
  d * e = -14 := by
  sorry

end quadratic_rewrite_l1114_111426


namespace expression_equality_l1114_111442

theorem expression_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_equality_l1114_111442


namespace failing_percentage_possible_l1114_111472

theorem failing_percentage_possible (n : ℕ) (d a : ℕ) (f_d f_a : ℕ) :
  n = 25 →
  d + a ≥ n →
  (f_d : ℚ) / (d : ℚ) = 3 / 10 →
  (f_a : ℚ) / (a : ℚ) = 3 / 10 →
  ∃ (f_total : ℕ), (f_total : ℚ) / (n : ℚ) > 7 / 20 ∧ f_total ≤ f_d + f_a :=
by sorry


end failing_percentage_possible_l1114_111472


namespace bicycle_wheel_radius_l1114_111474

theorem bicycle_wheel_radius (diameter : ℝ) (h : diameter = 26) : 
  diameter / 2 = 13 := by
  sorry

end bicycle_wheel_radius_l1114_111474


namespace three_balls_four_boxes_l1114_111412

theorem three_balls_four_boxes :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 4
  num_boxes ^ num_balls = 64 :=
by sorry

end three_balls_four_boxes_l1114_111412


namespace exists_non_monochromatic_coloring_l1114_111414

/-- Represents a coloring of numbers using 4 colors -/
def Coloring := Fin 2008 → Fin 4

/-- An arithmetic progression of 10 terms -/
def ArithmeticProgression := Fin 10 → Fin 2008

/-- Checks if an arithmetic progression is valid (within the range 1 to 2008) -/
def isValidAP (ap : ArithmeticProgression) : Prop :=
  ∀ i : Fin 10, ap i < 2008

/-- Checks if an arithmetic progression is monochromatic under a given coloring -/
def isMonochromatic (c : Coloring) (ap : ArithmeticProgression) : Prop :=
  ∃ color : Fin 4, ∀ i : Fin 10, c (ap i) = color

/-- The main theorem statement -/
theorem exists_non_monochromatic_coloring :
  ∃ c : Coloring, ∀ ap : ArithmeticProgression, isValidAP ap → ¬isMonochromatic c ap := by
  sorry

end exists_non_monochromatic_coloring_l1114_111414


namespace student_group_size_l1114_111493

theorem student_group_size (n : ℕ) (h : n > 1) :
  (2 : ℚ) / n = (1 : ℚ) / 5 → n = 10 := by
  sorry

end student_group_size_l1114_111493


namespace plant_arrangement_count_l1114_111458

-- Define the number of each type of plant
def num_basil : ℕ := 3
def num_tomato : ℕ := 3
def num_pepper : ℕ := 2

-- Define the number of tomato groups
def num_tomato_groups : ℕ := 2

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  (Nat.factorial num_basil) *
  (Nat.choose num_tomato 2) *
  (Nat.factorial 2) *
  (Nat.choose (num_basil + 1) num_tomato_groups) *
  (Nat.factorial num_pepper)

-- Theorem statement
theorem plant_arrangement_count :
  num_arrangements = 432 :=
sorry

end plant_arrangement_count_l1114_111458


namespace product_equals_fraction_l1114_111429

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18 / 5 := by
  sorry

end product_equals_fraction_l1114_111429


namespace franks_decks_l1114_111467

theorem franks_decks (deck_cost : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  deck_cost = 7 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (frank_decks : ℕ), frank_decks * deck_cost + friend_decks * deck_cost = total_spent ∧ frank_decks = 3 :=
by sorry

end franks_decks_l1114_111467


namespace correct_order_of_operations_l1114_111487

-- Define the expression
def expression : List ℤ := [150, -50, 25, 5]

-- Define the operations
inductive Operation
| Addition
| Subtraction
| Multiplication

-- Define the order of operations
def orderOfOperations : List Operation := [Operation.Multiplication, Operation.Subtraction, Operation.Addition]

-- Function to evaluate the expression
def evaluate (expr : List ℤ) (ops : List Operation) : ℤ :=
  sorry

-- Theorem statement
theorem correct_order_of_operations :
  evaluate expression orderOfOperations = 225 :=
sorry

end correct_order_of_operations_l1114_111487


namespace kyles_weight_lifting_ratio_l1114_111410

/-- 
Given:
- Kyle can lift 60 more pounds this year
- He can now lift 80 pounds in total
Prove that the ratio of the additional weight to the weight he could lift last year is 3
-/
theorem kyles_weight_lifting_ratio : 
  ∀ (last_year_weight additional_weight total_weight : ℕ),
  additional_weight = 60 →
  total_weight = 80 →
  total_weight = last_year_weight + additional_weight →
  (additional_weight : ℚ) / last_year_weight = 3 := by
sorry

end kyles_weight_lifting_ratio_l1114_111410


namespace representatives_selection_count_l1114_111494

def male_students : ℕ := 6
def female_students : ℕ := 3
def total_students : ℕ := male_students + female_students
def representatives : ℕ := 4

theorem representatives_selection_count :
  (Nat.choose total_students representatives) - (Nat.choose male_students representatives) = 111 := by
  sorry

end representatives_selection_count_l1114_111494


namespace quadratic_equation_one_solution_l1114_111480

theorem quadratic_equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 + 2 * k * x + 1 = 0) ↔ (k = -2 ∨ k = -1 ∨ k = 2) :=
sorry

end quadratic_equation_one_solution_l1114_111480


namespace min_value_expression_min_value_achievable_l1114_111413

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) ≥ -1814505489.667 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) = -1814505489.667 :=
sorry

end min_value_expression_min_value_achievable_l1114_111413


namespace line_properties_l1114_111461

/-- Triangle PQR with vertices P(1, 9), Q(3, 2), and R(9, 2) -/
structure Triangle where
  P : ℝ × ℝ := (1, 9)
  Q : ℝ × ℝ := (3, 2)
  R : ℝ × ℝ := (9, 2)

/-- A line defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Function to calculate the area of a triangle given its vertices -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a line cuts the triangle's area in half -/
def cutsAreaInHalf (t : Triangle) (l : Line) : Prop := sorry

/-- Theorem stating the properties of the line that cuts the triangle's area in half -/
theorem line_properties (t : Triangle) (l : Line) :
  cutsAreaInHalf t l ∧ l.yIntercept = 1 →
  l.slope = 1/3 ∧ l.slope + l.yIntercept = 4/3 := by sorry

end line_properties_l1114_111461


namespace min_value_inequality_l1114_111437

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end min_value_inequality_l1114_111437


namespace trajectory_is_two_circles_l1114_111481

-- Define the set of complex numbers satisfying the equation
def S : Set ℂ := {z : ℂ | Complex.abs z ^ 2 - 3 * Complex.abs z + 2 = 0}

-- Define the trajectory of z
def trajectory (z : ℂ) : Set ℂ := {w : ℂ | Complex.abs w = Complex.abs z}

-- Theorem statement
theorem trajectory_is_two_circles :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
  (∀ z ∈ S, (trajectory z = {w : ℂ | Complex.abs w = r₁} ∨
             trajectory z = {w : ℂ | Complex.abs w = r₂})) :=
sorry

end trajectory_is_two_circles_l1114_111481


namespace fraction_reducibility_implies_divisibility_l1114_111456

theorem fraction_reducibility_implies_divisibility 
  (a b c n l p : ℤ) 
  (h_reducible : ∃ (k m : ℤ), a * l + b = p * k ∧ c * l + n = p * m) : 
  p ∣ (a * n - b * c) := by
sorry

end fraction_reducibility_implies_divisibility_l1114_111456


namespace total_pies_sold_is_29_l1114_111498

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_orders : ℕ := 52

/-- Represents the number of customers who ordered slices of chicken pot pie -/
def chicken_pot_pie_orders : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ :=
  shepherds_pie_orders / shepherds_pie_pieces +
  chicken_pot_pie_orders / chicken_pot_pie_pieces

/-- Proves that the total number of pies sold is 29 -/
theorem total_pies_sold_is_29 : total_pies_sold = 29 := by
  sorry

end total_pies_sold_is_29_l1114_111498


namespace min_value_expression_l1114_111471

theorem min_value_expression (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + b * c + c * a > 0)
  (h3 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (2 / |a - b| + 2 / |b - c| + 2 / |c - a| + 5 / Real.sqrt (a * b + b * c + c * a)) ≥ 10 * Real.sqrt 6 :=
by sorry

end min_value_expression_l1114_111471


namespace pizza_solution_l1114_111431

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  others : ℕ

def pizza_theorem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.small = 3 ∧
  purchased.large = 2 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = (eaten.bob + 1) / 2 ∧
  eaten.others = 9 ∧
  (slices.small * purchased.small + slices.large * purchased.large) - 
    (eaten.george + eaten.bob + eaten.susie + eaten.others) = 10 →
  eaten.george = 6

theorem pizza_solution : 
  ∃ (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten),
    pizza_theorem slices purchased eaten := by
  sorry

end pizza_solution_l1114_111431


namespace two_distinct_prime_products_count_l1114_111400

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of integers less than n that are the product of exactly two distinct primes -/
def countTwoDistinctPrimeProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of numbers less than 1,000,000 that are the product of exactly two distinct primes is 209867 -/
theorem two_distinct_prime_products_count :
  countTwoDistinctPrimeProducts 1000000 = 209867 := by sorry

end two_distinct_prime_products_count_l1114_111400


namespace algebraic_simplification_l1114_111415

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification (a b : ℝ) :
  (3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1) ∧
  (2 * (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = 10 * a - 3 * a^2) := by
  sorry

end algebraic_simplification_l1114_111415


namespace arithmetic_square_root_of_six_l1114_111460

theorem arithmetic_square_root_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end arithmetic_square_root_of_six_l1114_111460


namespace product_properties_l1114_111484

theorem product_properties (x y z : ℕ) : 
  x = 15 ∧ y = 5 ∧ z = 8 →
  (x * y * z = 600) ∧
  ((x - 10) * y * z = 200) ∧
  ((x + 5) * y * z = 1200) := by
sorry

end product_properties_l1114_111484


namespace x_minus_y_values_l1114_111455

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x + 1| = 4)
  (h2 : (y + 2)^2 = 4)
  (h3 : x + y ≥ -5) :
  (x - y = -5) ∨ (x - y = 3) ∨ (x - y = 7) :=
by sorry

end x_minus_y_values_l1114_111455


namespace divisibility_of_consecutive_ones_l1114_111407

/-- A number consisting of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_consecutive_ones :
  ∃ k : ℕ, consecutive_ones 1998 = 37 * k := by
  sorry

end divisibility_of_consecutive_ones_l1114_111407


namespace quadratic_inequality_condition_l1114_111408

theorem quadratic_inequality_condition (a : ℝ) (h : 0 ≤ a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by sorry

end quadratic_inequality_condition_l1114_111408


namespace point_inside_circle_l1114_111409

theorem point_inside_circle (a : ℝ) : 
  (∃ (x y : ℝ), x = 2*a ∧ y = a - 1 ∧ x^2 + y^2 - 2*y - 4 < 0) ↔ 
  (-1/5 < a ∧ a < 1) :=
sorry

end point_inside_circle_l1114_111409


namespace min_value_fraction_l1114_111438

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  4/x + 1/y ≥ 6 + 4*Real.sqrt 2 ∧
  (4/x + 1/y = 6 + 4*Real.sqrt 2 ↔ x = 2 - Real.sqrt 2 ∧ y = (Real.sqrt 2 - 1)/2) :=
sorry

end min_value_fraction_l1114_111438


namespace cubic_room_floor_perimeter_l1114_111488

/-- The perimeter of the floor of a cubic room -/
def floor_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of the floor of a cubic room with side length 5 meters is 20 meters -/
theorem cubic_room_floor_perimeter :
  floor_perimeter 5 = 20 := by
  sorry

end cubic_room_floor_perimeter_l1114_111488


namespace stewart_farm_sheep_count_l1114_111406

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    num_sheep / num_horses = 2 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 16 := by
  sorry

end stewart_farm_sheep_count_l1114_111406


namespace derivative_at_pi_half_l1114_111401

/-- Given a function f where f(x) = sin x + 2x * f'(0), prove that f'(π/2) = -2 -/
theorem derivative_at_pi_half (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + 2 * x * (deriv f 0)) :
  deriv f (π/2) = -2 := by
  sorry

end derivative_at_pi_half_l1114_111401


namespace ice_cream_flavors_count_l1114_111443

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := stars_and_bars 4 4

theorem ice_cream_flavors_count : ice_cream_flavors = 35 := by sorry

end ice_cream_flavors_count_l1114_111443


namespace certain_number_proof_l1114_111435

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x - (1/3) * (0.15 * x)) = 18 → x = 180 := by
  sorry

end certain_number_proof_l1114_111435


namespace quadratic_roots_problem_l1114_111447

theorem quadratic_roots_problem (a b c : ℤ) (h_prime : Prime (a + b + c)) :
  let f : ℤ → ℤ := λ x => a * x * x + b * x + c
  (∃ x y : ℕ, x ≠ y ∧ f x = 0 ∧ f y = 0) →  -- roots are distinct positive integers
  (∃ r : ℕ, f r = -55) →                    -- substituting one root gives -55
  (∃ x y : ℕ, x = 2 ∧ y = 7 ∧ f x = 0 ∧ f y = 0) :=
by sorry

end quadratic_roots_problem_l1114_111447


namespace eight_power_15_divided_by_64_power_6_l1114_111451

theorem eight_power_15_divided_by_64_power_6 : 8^15 / 64^6 = 512 := by sorry

end eight_power_15_divided_by_64_power_6_l1114_111451


namespace old_machine_rate_proof_l1114_111489

/-- The rate at which the new machine makes bolts (in bolts per hour) -/
def new_machine_rate : ℝ := 150

/-- The time both machines work together (in hours) -/
def work_time : ℝ := 2

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 500

/-- The rate at which the old machine makes bolts (in bolts per hour) -/
def old_machine_rate : ℝ := 100

theorem old_machine_rate_proof :
  old_machine_rate * work_time + new_machine_rate * work_time = total_bolts :=
by sorry

end old_machine_rate_proof_l1114_111489


namespace sin_translation_to_cos_l1114_111468

theorem sin_translation_to_cos (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6)
  g x = Real.cos (2 * x) := by
sorry

end sin_translation_to_cos_l1114_111468


namespace zero_det_necessary_not_sufficient_for_parallel_l1114_111478

/-- Represents a line in the Cartesian plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The determinant of the coefficients of two lines -/
def coeff_det (l₁ l₂ : Line) : ℝ :=
  l₁.a * l₂.b - l₂.a * l₁.b

/-- Theorem stating that zero determinant is necessary but not sufficient for parallel lines -/
theorem zero_det_necessary_not_sufficient_for_parallel (l₁ l₂ : Line) :
  (are_parallel l₁ l₂ → coeff_det l₁ l₂ = 0) ∧
  ¬(coeff_det l₁ l₂ = 0 → are_parallel l₁ l₂) :=
sorry

end zero_det_necessary_not_sufficient_for_parallel_l1114_111478


namespace quadratic_root_zero_l1114_111422

/-- A quadratic equation in x with parameter m, where one root is zero -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + x + m^2 + 3*m = 0

/-- The theorem stating that m = -3 for the given quadratic equation -/
theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, quadratic_equation m x) ∧ 
  (quadratic_equation m 0) ∧ 
  (m ≠ 0) → 
  m = -3 := by
  sorry

end quadratic_root_zero_l1114_111422


namespace invitations_per_package_l1114_111470

theorem invitations_per_package (friends : ℕ) (packs : ℕ) (h1 : friends = 10) (h2 : packs = 5) :
  friends / packs = 2 := by
sorry

end invitations_per_package_l1114_111470


namespace select_one_from_two_sets_l1114_111444

theorem select_one_from_two_sets (left_set right_set : Finset ℕ) 
  (h1 : left_set.card = 15) (h2 : right_set.card = 20) 
  (h3 : left_set ∩ right_set = ∅) : 
  (left_set ∪ right_set).card = 35 := by
  sorry

end select_one_from_two_sets_l1114_111444


namespace inverse_of_proposition_l1114_111411

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end inverse_of_proposition_l1114_111411


namespace hendecagon_diagonal_intersection_probability_l1114_111496

/-- A regular hendecagon is an 11-sided polygon -/
def RegularHendecagon : Nat := 11

/-- The number of diagonals in a regular hendecagon -/
def NumDiagonals : Nat := (RegularHendecagon.choose 2) - RegularHendecagon

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def WaysToChooseTwoDiagonals : Nat := NumDiagonals.choose 2

/-- The number of sets of 4 vertices that determine intersecting diagonals -/
def IntersectingDiagonalSets : Nat := RegularHendecagon.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def IntersectionProbability : Rat := IntersectingDiagonalSets / WaysToChooseTwoDiagonals

theorem hendecagon_diagonal_intersection_probability :
  IntersectionProbability = 165 / 473 := by
  sorry

end hendecagon_diagonal_intersection_probability_l1114_111496


namespace product_ABC_l1114_111448

theorem product_ABC (m : ℝ) : 
  let A := 4 * m
  let B := m - (1/4 : ℝ)
  let C := m + (1/4 : ℝ)
  A * B * C = 4 * m^3 - (1/4 : ℝ) * m :=
by sorry

end product_ABC_l1114_111448


namespace intersection_of_P_and_Q_l1114_111418

def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | x ≥ 3}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 3 ≤ x ∧ x < 4} := by sorry

end intersection_of_P_and_Q_l1114_111418


namespace puppy_food_bags_l1114_111479

/-- Calculates the number of bags of special dog food needed for a puppy's first year -/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_weight : ℕ := 5
  let first_period : ℕ := 60
  let first_period_daily_food : ℕ := 2
  let second_period_daily_food : ℕ := 4
  let first_period_total : ℕ := first_period * first_period_daily_food
  let second_period : ℕ := days_in_year - first_period
  let second_period_total : ℕ := second_period * second_period_daily_food
  let total_ounces : ℕ := first_period_total + second_period_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_weight - 1) / bag_weight

theorem puppy_food_bags : bags_needed = 17 := by
  sorry

end puppy_food_bags_l1114_111479


namespace triangle_area_l1114_111486

theorem triangle_area (A B C : Real) (h1 : A > B) (h2 : B > C) 
  (h3 : 2 * Real.cos (2 * B) - 8 * Real.cos B + 5 = 0)
  (h4 : Real.tan A + Real.tan C = 3 + Real.sqrt 3)
  (h5 : 2 * Real.sqrt 3 = Real.sin C * (A - C)) : 
  (1 / 2) * (A - C) * 2 * Real.sqrt 3 = 12 - 4 * Real.sqrt 3 := by
  sorry

#check triangle_area

end triangle_area_l1114_111486


namespace initial_average_calculation_l1114_111475

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) : 
  n = 10 → 
  correct_avg = 6 → 
  error = 10 → 
  (n * correct_avg - error) / n = 5 := by
sorry

end initial_average_calculation_l1114_111475


namespace tan_c_in_triangle_l1114_111445

theorem tan_c_in_triangle (A B C : Real) : 
  -- Triangle condition
  A + B + C = π → 
  -- tan A and tan B are roots of 3x^2 - 7x + 2 = 0
  (∃ (x y : Real), x ≠ y ∧ 
    3 * x^2 - 7 * x + 2 = 0 ∧ 
    3 * y^2 - 7 * y + 2 = 0 ∧ 
    x = Real.tan A ∧ 
    y = Real.tan B) → 
  -- Conclusion
  Real.tan C = -7 :=
by sorry

end tan_c_in_triangle_l1114_111445


namespace point_between_parallel_lines_l1114_111454

theorem point_between_parallel_lines :
  ∃ (b : ℤ),
    (31 - 8 * b) * (20 - 4 * b) < 0 ∧
    b = 4 :=
by sorry

end point_between_parallel_lines_l1114_111454


namespace valid_grid_exists_l1114_111434

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers are adjacent in the grid -/
def adjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The main theorem stating the existence of a valid grid -/
theorem valid_grid_exists : ∃ (g : Grid),
  (∀ i j i' j', adjacent i j i' j' → (g i j ∣ g i' j' ∨ g i' j' ∣ g i j)) ∧
  (∀ i j, g i j ≤ 25) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → g i j ≠ g i' j') :=
sorry

end valid_grid_exists_l1114_111434


namespace D_sqrt_sometimes_rational_sometimes_not_l1114_111497

def D (x : ℝ) : ℝ := 
  let a := 2*x + 1
  let b := 2*x + 3
  let c := a*b + 5
  a^2 + b^2 + c^2

theorem D_sqrt_sometimes_rational_sometimes_not :
  ∃ x y : ℝ, (∃ q : ℚ, Real.sqrt (D x) = q) ∧ 
             (∀ q : ℚ, Real.sqrt (D y) ≠ q) :=
sorry

end D_sqrt_sometimes_rational_sometimes_not_l1114_111497


namespace rectangle_area_l1114_111425

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  w ^ 2 + l ^ 2 = x ^ 2 ∧
  w * l = (3 / 10) * x ^ 2 := by
  sorry

end rectangle_area_l1114_111425


namespace count_divisible_by_four_l1114_111402

theorem count_divisible_by_four : 
  (Finset.filter (fun n : Fin 10 => (748 * 10 + n : ℕ) % 4 = 0) Finset.univ).card = 3 :=
by sorry

end count_divisible_by_four_l1114_111402


namespace magnitude_of_sum_l1114_111416

def a (m : ℝ) : ℝ × ℝ := (4, m)
def b : ℝ × ℝ := (1, -2)

theorem magnitude_of_sum (m : ℝ) 
  (h : (a m).1 * b.1 + (a m).2 * b.2 = 0) : 
  Real.sqrt (((a m).1 + 2 * b.1)^2 + ((a m).2 + 2 * b.2)^2) = 2 * Real.sqrt 10 := by
  sorry

end magnitude_of_sum_l1114_111416


namespace function_minimum_and_inequality_l1114_111427

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |5 - x|

-- State the theorem
theorem function_minimum_and_inequality :
  ∃ (m : ℝ), 
    (∀ x, f x ≥ m) ∧ 
    (∃ x, f x = m) ∧
    m = 9/2 ∧
    ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = (2/3) * m → 
      1 / (a + 1) + 1 / (b + 2) ≥ 2/3 :=
by sorry

end function_minimum_and_inequality_l1114_111427


namespace cubic_sum_divisible_by_nine_l1114_111465

theorem cubic_sum_divisible_by_nine (n : ℕ+) : 
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end cubic_sum_divisible_by_nine_l1114_111465


namespace diagonals_25_sided_polygon_l1114_111491

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end diagonals_25_sided_polygon_l1114_111491


namespace initial_amount_theorem_l1114_111463

/-- The amount of money in Olivia's wallet before visiting the supermarket. -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket. -/
def amount_spent : ℕ := 16

/-- The amount of money left in Olivia's wallet after visiting the supermarket. -/
def amount_left : ℕ := 78

/-- Theorem stating that the initial amount in Olivia's wallet was $94. -/
theorem initial_amount_theorem : initial_amount = 94 :=
by
  sorry

end initial_amount_theorem_l1114_111463


namespace intersection_x_is_seven_l1114_111417

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Point C -/
def C : Point := ⟨7, 5⟩

/-- Point D -/
def D : Point := ⟨7, -3⟩

/-- The line passing through points C and D -/
def line_CD : Line := { p1 := C, p2 := D }

/-- The x-coordinate of the intersection point between a line and the x-axis -/
def intersection_x (l : Line) : ℝ := sorry

theorem intersection_x_is_seven : intersection_x line_CD = 7 := by sorry

end intersection_x_is_seven_l1114_111417


namespace triangle_similarity_properties_l1114_111485

/-- Triangle properties for medians and altitudes similarity --/
theorem triangle_similarity_properties (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) :
  (∀ (ma mb mc : ℝ), 
    (ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)) →
    (mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)) →
    (mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) →
    (a / mc = b / mb ∧ b / mb = c / ma) →
    (2 * b^2 = a^2 + c^2)) ∧
  (∀ (ha hb hc : ℝ),
    (ha * a = hb * b ∧ hb * b = hc * c) →
    (ha / hb = b / a ∧ ha / hc = c / a ∧ hb / hc = c / b)) := by
  sorry


end triangle_similarity_properties_l1114_111485


namespace tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l1114_111419

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x + 1

-- Theorem for the tangent line when a = 0
theorem tangent_line_a_zero :
  ∀ x y : ℝ, f 0 1 = 1 → (2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1)) := by sorry

-- Theorem for the maximum value of g when a > 0
theorem max_value_g_positive_a :
  ∀ a : ℝ, a > 0 → ∃ max_val : ℝ, max_val = g a (1/a) ∧ 
  ∀ x : ℝ, x > 0 → g a x ≤ max_val := by sorry

-- Theorem for the inequality when a = -2
theorem inequality_a_negative_two :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
  f (-2) x₁ + f (-2) x₂ + x₁ * x₂ = 0 → 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by sorry

end

end tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l1114_111419


namespace count_cubes_with_at_most_two_shared_vertices_l1114_111464

/-- Given a cube with edge length n divided into n^3 unit cubes, 
    this function calculates the number of unit cubes that share 
    no more than 2 vertices with any other unit cube. -/
def cubes_with_at_most_two_shared_vertices (n : ℕ) : ℕ :=
  (n^2 * (n^4 - 7*n + 6)) / 2

/-- Theorem stating that the number of unit cubes sharing no more than 2 vertices 
    in a cube of edge length n is given by the formula (1/2) * n^2 * (n^4 - 7n + 6). -/
theorem count_cubes_with_at_most_two_shared_vertices (n : ℕ) :
  cubes_with_at_most_two_shared_vertices n = (n^2 * (n^4 - 7*n + 6)) / 2 :=
by sorry

end count_cubes_with_at_most_two_shared_vertices_l1114_111464


namespace bus_speed_excluding_stoppages_l1114_111441

/-- Given a bus that travels at 43 kmph including stoppages and stops for 8.4 minutes per hour,
    its speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages (speed_with_stops : ℝ) (stoppage_time : ℝ) :
  speed_with_stops = 43 →
  stoppage_time = 8.4 →
  (60 - stoppage_time) / 60 * speed_with_stops = 50 := by
  sorry

#check bus_speed_excluding_stoppages

end bus_speed_excluding_stoppages_l1114_111441


namespace tree_height_problem_l1114_111423

/-- Given a square ABCD with trees of heights a, b, c at vertices A, B, C respectively,
    and a point O inside the square equidistant from all vertices,
    prove that the height of the tree at vertex D is √(a² + c² - b²). -/
theorem tree_height_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = a^2 + c^2 - b^2 := by
  sorry


end tree_height_problem_l1114_111423


namespace apartment_complex_households_l1114_111452

/-- The maximum number of households in Jungkook's apartment complex -/
def max_households : ℕ := 2000

/-- The maximum number of buildings in the apartment complex -/
def max_buildings : ℕ := 25

/-- The maximum number of floors per building -/
def max_floors : ℕ := 10

/-- The number of households per floor -/
def households_per_floor : ℕ := 8

/-- Theorem stating that the maximum number of households in the apartment complex is 2000 -/
theorem apartment_complex_households :
  max_households = max_buildings * max_floors * households_per_floor :=
by sorry

end apartment_complex_households_l1114_111452


namespace chalkboard_area_l1114_111482

/-- The area of a rectangular chalkboard with width 3 feet and length 2 times its width is 18 square feet. -/
theorem chalkboard_area (width : ℝ) (length : ℝ) : 
  width = 3 → length = 2 * width → width * length = 18 := by
  sorry

end chalkboard_area_l1114_111482


namespace min_value_implies_m_l1114_111477

/-- The function f(x) = -x^3 + 6x^2 + m -/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 + m

/-- Theorem: If f(x) has a minimum value of 23, then m = 23 -/
theorem min_value_implies_m (m : ℝ) : 
  (∃ (y : ℝ), ∀ (x : ℝ), f x m ≥ y ∧ ∃ (x₀ : ℝ), f x₀ m = y) ∧ 
  (∃ (x₀ : ℝ), f x₀ m = 23) → 
  m = 23 := by
  sorry


end min_value_implies_m_l1114_111477


namespace cost_reduction_proof_l1114_111492

theorem cost_reduction_proof (x : ℝ) : 
  (x ≥ 0) →  -- Ensure x is non-negative
  (x ≤ 1) →  -- Ensure x is at most 100%
  ((1 - x)^2 = 1 - 0.36) →
  x = 0.2 :=
by sorry

end cost_reduction_proof_l1114_111492


namespace same_end_word_count_l1114_111403

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A four-letter word with the same first and last letter -/
structure SameEndWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The count of all possible SameEndWords -/
def count_same_end_words : ℕ := alphabet_size * alphabet_size * alphabet_size

theorem same_end_word_count :
  count_same_end_words = 17576 :=
sorry

end same_end_word_count_l1114_111403
