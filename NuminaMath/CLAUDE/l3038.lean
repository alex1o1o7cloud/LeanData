import Mathlib

namespace batsman_average_after_17th_innings_l3038_303820

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (score : ℕ) : ℚ :=
  (b.totalScore + score) / (b.innings + 1)

/-- Theorem stating the batsman's new average after the 17th innings -/
theorem batsman_average_after_17th_innings
  (b : Batsman)
  (h1 : b.innings = 16)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 37 := by
  sorry

#check batsman_average_after_17th_innings

end batsman_average_after_17th_innings_l3038_303820


namespace f_has_max_and_min_l3038_303862

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 6

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- Theorem stating the condition for f to have both maximum and minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ x y : ℝ, ∀ z : ℝ, f a z ≤ f a x ∧ f a y ≤ f a z) ↔ a < 1/3 ∧ a ≠ 0 :=
sorry

end f_has_max_and_min_l3038_303862


namespace unique_solution_l3038_303893

theorem unique_solution (m n : ℕ+) 
  (eq : 2 * m.val + 3 = 5 * n.val - 2)
  (ineq : 5 * n.val - 2 < 15) :
  m.val = 5 ∧ n.val = 3 := by
sorry

end unique_solution_l3038_303893


namespace geometric_sequence_sum_l3038_303819

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 12 + a 3 * a 5 = 15 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = 5 := by
  sorry

end geometric_sequence_sum_l3038_303819


namespace num_machines_is_five_l3038_303830

/-- The number of machines in the first scenario -/
def num_machines : ℕ := 5

/-- The production rate of the machines in the first scenario -/
def production_rate_1 : ℚ := 20 / (10 * num_machines)

/-- The production rate of the machines in the second scenario -/
def production_rate_2 : ℚ := 200 / (25 * 20)

/-- Theorem stating that the number of machines in the first scenario is 5 -/
theorem num_machines_is_five :
  num_machines = 5 ∧ production_rate_1 = production_rate_2 :=
sorry

end num_machines_is_five_l3038_303830


namespace second_project_depth_l3038_303835

/-- Represents the dimensions of a digging project -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ :=
  p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject :=
  { depth := 100, length := 25, breadth := 30 }

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject :=
  { depth := depth, length := 20, breadth := 50 }

theorem second_project_depth :
  ∃ d : ℝ, volume project1 = volume (project2 d) ∧ d = 75 := by
  sorry

end second_project_depth_l3038_303835


namespace arithmetic_less_than_geometric_l3038_303864

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), d > 0 ∧ ∀ n, a n = a₁ + (n - 1) * d

/-- A positive geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r > 1 ∧ ∀ n, b n = b₁ * r^(n - 1)

theorem arithmetic_less_than_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end arithmetic_less_than_geometric_l3038_303864


namespace election_votes_theorem_l3038_303860

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : ∃ (winner_votes loser_votes : ℕ), 
    winner_votes + loser_votes = total_votes ∧ 
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 192) :
  total_votes = 480 := by
sorry

end election_votes_theorem_l3038_303860


namespace fraction_equals_zero_l3038_303811

theorem fraction_equals_zero (x : ℝ) : (x + 1) / (x - 2) = 0 → x = -1 := by
  sorry

end fraction_equals_zero_l3038_303811


namespace subset_sum_divisible_by_p_l3038_303885

theorem subset_sum_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (S.powerset.filter (fun A => A.card = p ∧ (A.sum id) % p = 0)).card =
    (Nat.choose (2 * p) p - 2) / p + 2 := by
  sorry

end subset_sum_divisible_by_p_l3038_303885


namespace initial_solution_volume_l3038_303855

/-- Proves that the initial amount of solution is 6 litres, given that it is 25% alcohol
    and becomes 50% alcohol when 3 litres of pure alcohol are added. -/
theorem initial_solution_volume (x : ℝ) :
  (0.25 * x) / x = 0.25 →
  ((0.25 * x + 3) / (x + 3) = 0.5) →
  x = 6 := by
sorry

end initial_solution_volume_l3038_303855


namespace cut_rectangle_properties_l3038_303876

/-- Represents a rectangle cut into four pieces by two equal diagonals intersecting at right angles -/
structure CutRectangle where
  width : ℝ
  height : ℝ
  diag_intersect_center : Bool
  diag_right_angle : Bool
  diag_equal_length : Bool

/-- Theorem about properties of a specific cut rectangle -/
theorem cut_rectangle_properties (rect : CutRectangle) 
  (h_width : rect.width = 20)
  (h_height : rect.height = 30)
  (h_center : rect.diag_intersect_center = true)
  (h_right : rect.diag_right_angle = true)
  (h_equal : rect.diag_equal_length = true) :
  ∃ (square_side triangle_area pentagon_area hole_area : ℝ),
    square_side = 20 ∧
    triangle_area = 100 ∧
    pentagon_area = 200 ∧
    hole_area = 200 := by
  sorry


end cut_rectangle_properties_l3038_303876


namespace exterior_angle_sum_l3038_303880

/-- In a triangle ABC, the exterior angle α at vertex A is equal to the sum of the two non-adjacent interior angles B and C. -/
theorem exterior_angle_sum (A B C : Real) (α : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → α = B + C :=
by sorry

end exterior_angle_sum_l3038_303880


namespace cookies_remaining_l3038_303896

/-- Given the initial number of cookies and the number of cookies taken by each person,
    prove that the remaining number of cookies is 6. -/
theorem cookies_remaining (initial : ℕ) (eaten : ℕ) (brother : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ)
    (h1 : initial = 22)
    (h2 : eaten = 2)
    (h3 : brother = 1)
    (h4 : friend1 = 3)
    (h5 : friend2 = 5)
    (h6 : friend3 = 5) :
    initial - eaten - brother - friend1 - friend2 - friend3 = 6 := by
  sorry

end cookies_remaining_l3038_303896


namespace oscar_review_questions_l3038_303872

/-- The total number of questions Professor Oscar must review -/
def total_questions (questions_per_exam : ℕ) (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  questions_per_exam * num_classes * students_per_class

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 10 5 35 = 1750 := by
  sorry

end oscar_review_questions_l3038_303872


namespace equation_solutions_l3038_303871

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 9 = 3 * x - 16 ∧ x = -7/2) ∧
  (∃ x : ℚ, (3 * x - 1) / 3 = 1 - (x + 2) / 4 ∧ x = 2/3) := by
  sorry

end equation_solutions_l3038_303871


namespace house_wall_nails_l3038_303892

/-- The number of nails needed for large planks -/
def large_planks_nails : ℕ := 15

/-- The number of nails needed for small planks -/
def small_planks_nails : ℕ := 5

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := large_planks_nails + small_planks_nails

theorem house_wall_nails : total_nails = 20 := by
  sorry

end house_wall_nails_l3038_303892


namespace room_volume_l3038_303873

/-- Given a room with length three times its breadth, height twice its breadth,
    and floor area of 12 sq.m, prove that its volume is 48 cubic meters. -/
theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 := by
  sorry

end room_volume_l3038_303873


namespace cube_root_of_64_l3038_303802

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end cube_root_of_64_l3038_303802


namespace correct_operation_l3038_303821

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end correct_operation_l3038_303821


namespace least_whole_number_subtraction_l3038_303803

-- Define the original ratio
def original_ratio : Rat := 6 / 7

-- Define the comparison ratio
def comparison_ratio : Rat := 16 / 21

-- Define the function that creates the new ratio after subtracting x
def new_ratio (x : ℕ) : Rat := (6 - x) / (7 - x)

-- Statement to prove
theorem least_whole_number_subtraction :
  ∀ x : ℕ, x < 3 → new_ratio x ≥ comparison_ratio ∧
  new_ratio 3 < comparison_ratio :=
by sorry

end least_whole_number_subtraction_l3038_303803


namespace intersection_of_A_and_B_l3038_303829

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l3038_303829


namespace expression_evaluation_l3038_303887

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l3038_303887


namespace miles_difference_l3038_303827

/-- The number of miles Gervais drove per day -/
def gervais_daily_miles : ℕ := 315

/-- The number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- The total number of miles Henri drove -/
def henri_total_miles : ℕ := 1250

/-- Theorem stating the difference in miles driven between Henri and Gervais -/
theorem miles_difference : henri_total_miles - (gervais_daily_miles * gervais_days) = 305 := by
  sorry

end miles_difference_l3038_303827


namespace quadratic_completing_square_l3038_303813

/-- Given a quadratic equation 16x^2 - 32x - 512 = 0, when transformed
    to the form (x + p)^2 = q, the value of q is 33. -/
theorem quadratic_completing_square :
  ∃ (p : ℝ), ∀ (x : ℝ),
    16 * x^2 - 32 * x - 512 = 0 ↔ (x + p)^2 = 33 :=
by sorry

end quadratic_completing_square_l3038_303813


namespace negation_of_universal_proposition_l3038_303889

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0) := by sorry

end negation_of_universal_proposition_l3038_303889


namespace quadratic_function_max_value_l3038_303826

theorem quadratic_function_max_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a = 2) → 
  a = -1 ∨ a = 2 := by
sorry

end quadratic_function_max_value_l3038_303826


namespace solution_to_equation_l3038_303836

theorem solution_to_equation : ∃! x : ℝ, (x - 3)^3 = (1/27)⁻¹ := by
  use 6
  sorry

end solution_to_equation_l3038_303836


namespace complement_union_theorem_l3038_303817

open Set

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,3,5}
def Q : Set Nat := {1,2,4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1,2,4,6} := by sorry

end complement_union_theorem_l3038_303817


namespace bicycle_helmet_cost_ratio_l3038_303818

theorem bicycle_helmet_cost_ratio :
  ∀ (bicycle_cost helmet_cost : ℕ),
    helmet_cost = 40 →
    bicycle_cost + helmet_cost = 240 →
    ∃ (m : ℕ), bicycle_cost = m * helmet_cost →
    m = 5 := by
  sorry

end bicycle_helmet_cost_ratio_l3038_303818


namespace gcd_of_638_522_406_l3038_303856

theorem gcd_of_638_522_406 : Nat.gcd 638 (Nat.gcd 522 406) = 2 := by
  sorry

end gcd_of_638_522_406_l3038_303856


namespace sum_of_fourth_and_fifth_terms_l3038_303898

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem sum_of_fourth_and_fifth_terms :
  let a₁ : ℝ := 4096
  let r : ℝ := 1/4
  (geometric_sequence a₁ r 4) + (geometric_sequence a₁ r 5) = 80 := by
  sorry

end sum_of_fourth_and_fifth_terms_l3038_303898


namespace solution_set_l3038_303883

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (3 - m, 1)

-- Define the condition for P being in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2 - m) * x + m > 2

theorem solution_set (m : ℝ) : 
  is_in_second_quadrant (P m) → 
  (∀ x : ℝ, inequality m x ↔ x < 1) :=
sorry

end solution_set_l3038_303883


namespace cylinder_unique_non_trapezoid_cross_section_l3038_303840

-- Define the solids
inductive Solid
| Frustum
| Cylinder
| Cube
| TriangularPrism

-- Define a predicate for whether a solid can have an isosceles trapezoid cross-section
def can_have_isosceles_trapezoid_cross_section : Solid → Prop
| Solid.Frustum => True
| Solid.Cylinder => False
| Solid.Cube => True
| Solid.TriangularPrism => True

-- Theorem statement
theorem cylinder_unique_non_trapezoid_cross_section :
  ∀ s : Solid, ¬(can_have_isosceles_trapezoid_cross_section s) ↔ s = Solid.Cylinder :=
by sorry

end cylinder_unique_non_trapezoid_cross_section_l3038_303840


namespace nancy_chips_to_brother_l3038_303879

def tortilla_chips_problem (total_chips : ℕ) (kept_chips : ℕ) (sister_chips : ℕ) : ℕ :=
  total_chips - kept_chips - sister_chips

theorem nancy_chips_to_brother :
  tortilla_chips_problem 22 10 5 = 7 := by
  sorry

end nancy_chips_to_brother_l3038_303879


namespace inequality_proof_l3038_303841

theorem inequality_proof (a b c d : ℝ) (h1 : c < d) (h2 : a > b) (h3 : b > 0) :
  a - c > b - d := by
  sorry

end inequality_proof_l3038_303841


namespace second_divisor_existence_l3038_303843

theorem second_divisor_existence : ∃ (D : ℕ+), 
  (∃ (N : ℤ), N % 35 = 25 ∧ N % D.val = 4) ∧ D.val = 21 := by
  sorry

end second_divisor_existence_l3038_303843


namespace bike_to_tractor_speed_ratio_l3038_303833

/-- Prove that the ratio of bike speed to tractor speed is 2:1 -/
theorem bike_to_tractor_speed_ratio :
  let tractor_speed := 575 / 23
  let car_speed := 360 / 4
  let bike_speed := car_speed / (9/5)
  bike_speed / tractor_speed = 2 := by
  sorry

end bike_to_tractor_speed_ratio_l3038_303833


namespace will_money_left_l3038_303851

/-- The amount of money Will has left after shopping --/
def money_left (initial_amount : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (shoes_price : ℝ) 
  (hat_price : ℝ) (socks_price : ℝ) (shoe_refund_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let new_total := total_cost - refund
  let remaining_items_cost := sweater_price + tshirt_price + hat_price + socks_price
  let discount := remaining_items_cost * discount_rate
  let discounted_total := new_total - discount
  let sales_tax := discounted_total * tax_rate
  let final_cost := discounted_total + sales_tax
  initial_amount - final_cost

/-- Theorem stating that Will has $41.87 left after shopping --/
theorem will_money_left : 
  money_left 74 9 11 30 5 4 0.85 0.1 0.05 = 41.87 := by
  sorry

end will_money_left_l3038_303851


namespace at_most_one_perfect_square_l3038_303807

def sequence_a : ℕ → ℕ
  | 0 => 1  -- arbitrary starting value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ k : ℕ, ∀ n m : ℕ, 
    (∃ i : ℕ, sequence_a n = i^2) → 
    (∃ j : ℕ, sequence_a m = j^2) → 
    n = m ∨ (n < k ∧ m < k) := by
  sorry

end at_most_one_perfect_square_l3038_303807


namespace greatest_prime_factor_of_factorial_sum_l3038_303845

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end greatest_prime_factor_of_factorial_sum_l3038_303845


namespace colored_paper_problem_l3038_303808

/-- The number of pieces of colored paper Yuna had initially -/
def yunas_initial_paper : ℕ := 150

/-- The number of pieces of colored paper Namjoon had initially -/
def namjoons_initial_paper : ℕ := 250

/-- The number of pieces of colored paper Namjoon gave to Yuna -/
def paper_given : ℕ := 60

/-- The difference in paper count between Yuna and Namjoon after the exchange -/
def paper_difference : ℕ := 20

theorem colored_paper_problem :
  yunas_initial_paper = 150 ∧
  namjoons_initial_paper = 250 ∧
  paper_given = 60 ∧
  paper_difference = 20 →
  yunas_initial_paper + paper_given = namjoons_initial_paper - paper_given + paper_difference :=
by sorry

end colored_paper_problem_l3038_303808


namespace weight_after_one_year_l3038_303857

def initial_weight : ℕ := 250

def training_loss : List ℕ := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]

def diet_loss_per_month : ℕ := 3

def months_in_year : ℕ := 12

theorem weight_after_one_year :
  initial_weight - (training_loss.sum + diet_loss_per_month * months_in_year) = 139 := by
  sorry

end weight_after_one_year_l3038_303857


namespace shepherd_a_has_seven_sheep_l3038_303805

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℕ
  b : ℕ

/-- The conditions of the problem are satisfied -/
def satisfiesConditions (s : ShepherdSheep) : Prop :=
  (s.a + 1 = 2 * (s.b - 1)) ∧ (s.a - 1 = s.b + 1)

/-- Theorem stating that shepherd A has 7 sheep -/
theorem shepherd_a_has_seven_sheep :
  ∃ s : ShepherdSheep, satisfiesConditions s ∧ s.a = 7 :=
sorry

end shepherd_a_has_seven_sheep_l3038_303805


namespace expression_evaluation_l3038_303891

theorem expression_evaluation (a b c : ℚ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

#eval (14 : ℚ) + 19 + 23

end expression_evaluation_l3038_303891


namespace youngest_son_cotton_correct_l3038_303868

/-- The amount of cotton for the youngest son in the "Dividing Cotton among Eight Sons" problem -/
def youngest_son_cotton : ℕ := 184

/-- The total amount of cotton to be divided -/
def total_cotton : ℕ := 996

/-- The number of sons -/
def num_sons : ℕ := 8

/-- The difference in cotton amount between each son -/
def cotton_difference : ℕ := 17

/-- Theorem stating that the youngest son's cotton amount is correct given the problem conditions -/
theorem youngest_son_cotton_correct :
  youngest_son_cotton * num_sons + (num_sons * (num_sons - 1) / 2) * cotton_difference = total_cotton :=
by sorry

end youngest_son_cotton_correct_l3038_303868


namespace cos_power_sum_l3038_303812

theorem cos_power_sum (α : ℝ) (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  x + 1/x = 2 * Real.cos α → x^n + 1/x^n = 2 * Real.cos (n * α) := by
  sorry

end cos_power_sum_l3038_303812


namespace chinese_riddle_championship_arrangement_l3038_303877

theorem chinese_riddle_championship_arrangement (n : ℕ) (students : ℕ) (teacher : ℕ) (parents : ℕ) :
  n = 6 →
  students = 3 →
  teacher = 1 →
  parents = 2 →
  (students.factorial * 2 * (n - students - 1).factorial) = 72 :=
by sorry

end chinese_riddle_championship_arrangement_l3038_303877


namespace solve_exponential_equation_l3038_303861

theorem solve_exponential_equation :
  ∃ n : ℕ, (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n = (64 : ℝ)^4 ∧ n = 2 := by
  sorry

end solve_exponential_equation_l3038_303861


namespace base5_413_equals_108_l3038_303869

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5^1 + c * 5^0

/-- The base 5 number 413₅ is equal to 108 in base 10 --/
theorem base5_413_equals_108 : base5ToBase10 4 1 3 = 108 := by sorry

end base5_413_equals_108_l3038_303869


namespace zero_not_in_range_of_g_l3038_303867

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end zero_not_in_range_of_g_l3038_303867


namespace constant_function_m_values_l3038_303899

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

-- State the theorem
theorem constant_function_m_values
  (m : ℝ)
  (h_exists : ∃ (a b : ℝ), -2 ≤ a ∧ a < b ∧
    ∀ x, x ∈ Set.Icc a b → ∃ c, f m x = c) :
  m = 1 ∨ m = -1 := by
sorry

end constant_function_m_values_l3038_303899


namespace id_tag_problem_l3038_303895

/-- The set of characters available for creating ID tags -/
def tag_chars : Finset Char := {'M', 'A', 'T', 'H', '2', '0', '3'}

/-- The number of times '2' can appear in a tag -/
def max_twos : Nat := 2

/-- The length of each ID tag -/
def tag_length : Nat := 5

/-- The total number of unique ID tags -/
def total_tags : Nat := 3720

/-- Theorem stating the result of the ID tag problem -/
theorem id_tag_problem :
  (total_tags : ℚ) / 10 = 372 := by sorry

end id_tag_problem_l3038_303895


namespace china_students_reading_l3038_303809

/-- Represents how a number is read in words -/
def NumberInWords : Type := String

/-- The correct way to read a given number -/
def correctReading (n : Float) : NumberInWords := sorry

/-- The number of primary school students enrolled in China in 2004 (in millions) -/
def chinaStudents2004 : Float := 11246.23

theorem china_students_reading :
  correctReading chinaStudents2004 = "eleven thousand two hundred forty-six point two three" := by
  sorry

end china_students_reading_l3038_303809


namespace line_equation_with_x_intercept_and_slope_angle_l3038_303890

theorem line_equation_with_x_intercept_and_slope_angle 
  (x_intercept : ℝ) 
  (slope_angle : ℝ) 
  (h1 : x_intercept = 2) 
  (h2 : slope_angle = 135) :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b ∧ 
    (x = x_intercept ∧ y = 0) ∧ 
    m = Real.tan (π - slope_angle * π / 180) ∧
    y = -x + 2 :=
by sorry

end line_equation_with_x_intercept_and_slope_angle_l3038_303890


namespace crow_votes_l3038_303859

def singing_contest (total_judges reported_total : ℕ)
                    (rooster_crow reported_rooster_crow : ℕ)
                    (crow_cuckoo reported_crow_cuckoo : ℕ)
                    (cuckoo_rooster reported_cuckoo_rooster : ℕ)
                    (max_error : ℕ) : Prop :=
  ∃ (rooster crow cuckoo : ℕ),
    -- Actual total of judges
    rooster + crow + cuckoo = total_judges ∧
    -- Reported total within error range
    (reported_total : ℤ) - (total_judges : ℤ) ≤ max_error ∧
    (total_judges : ℤ) - (reported_total : ℤ) ≤ max_error ∧
    -- Reported sums within error range
    (reported_rooster_crow : ℤ) - ((rooster + crow) : ℤ) ≤ max_error ∧
    ((rooster + crow) : ℤ) - (reported_rooster_crow : ℤ) ≤ max_error ∧
    (reported_crow_cuckoo : ℤ) - ((crow + cuckoo) : ℤ) ≤ max_error ∧
    ((crow + cuckoo) : ℤ) - (reported_crow_cuckoo : ℤ) ≤ max_error ∧
    (reported_cuckoo_rooster : ℤ) - ((cuckoo + rooster) : ℤ) ≤ max_error ∧
    ((cuckoo + rooster) : ℤ) - (reported_cuckoo_rooster : ℤ) ≤ max_error ∧
    -- The number of votes for Crow is 13
    crow = 13

theorem crow_votes :
  singing_contest 46 59 15 15 18 18 20 20 13 :=
by sorry

end crow_votes_l3038_303859


namespace absolute_value_equation_unique_solution_l3038_303837

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_unique_solution_l3038_303837


namespace min_sum_of_distances_l3038_303806

-- Define the curve
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the distance from a point to line y = 2
def dist_to_line1 (x y : ℝ) : ℝ := |y - 2|

-- Define the distance from a point to line x = -1
def dist_to_line2 (x y : ℝ) : ℝ := |x + 1|

-- Define the sum of distances
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 x y + dist_to_line2 x y

-- Theorem statement
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  (∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min) ∧
  (∃ (x y : ℝ), curve x y ∧ sum_of_distances x y = min) :=
sorry

end min_sum_of_distances_l3038_303806


namespace gcd_18_30_l3038_303816

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l3038_303816


namespace smallest_number_with_remainders_l3038_303832

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x % 4 = 1 ∧ 
  x % 3 = 2 ∧ 
  x % 5 = 3 ∧ 
  ∀ y : ℕ, (y % 4 = 1 ∧ y % 3 = 2 ∧ y % 5 = 3) → x ≤ y :=
by sorry

end smallest_number_with_remainders_l3038_303832


namespace quadratic_equation_solution_l3038_303866

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 2 := by
sorry

end quadratic_equation_solution_l3038_303866


namespace john_average_increase_l3038_303815

def john_scores : List ℝ := [92, 85, 91, 95]

theorem john_average_increase :
  let first_three_avg := (john_scores.take 3).sum / 3
  let all_four_avg := john_scores.sum / 4
  all_four_avg - first_three_avg = 1.42 := by sorry

end john_average_increase_l3038_303815


namespace shots_per_puppy_l3038_303831

/-- Calculates the number of shots each puppy needs given the specified conditions -/
theorem shots_per_puppy
  (num_dogs : ℕ)
  (puppies_per_dog : ℕ)
  (cost_per_shot : ℕ)
  (total_cost : ℕ)
  (h1 : num_dogs = 3)
  (h2 : puppies_per_dog = 4)
  (h3 : cost_per_shot = 5)
  (h4 : total_cost = 120) :
  (total_cost / cost_per_shot) / (num_dogs * puppies_per_dog) = 2 := by
  sorry

#check shots_per_puppy

end shots_per_puppy_l3038_303831


namespace beef_price_per_pound_l3038_303888

/-- The price of beef per pound given the total cost, number of packs, and weight per pack -/
def price_per_pound (total_cost : ℚ) (num_packs : ℕ) (weight_per_pack : ℚ) : ℚ :=
  total_cost / (num_packs * weight_per_pack)

/-- Theorem: The price of beef per pound is $5.50 -/
theorem beef_price_per_pound :
  price_per_pound 110 5 4 = 5.5 := by
  sorry

end beef_price_per_pound_l3038_303888


namespace emily_weight_l3038_303828

def heather_weight : ℕ := 87
def weight_difference : ℕ := 78

theorem emily_weight : 
  ∃ (emily_weight : ℕ), 
    emily_weight = heather_weight - weight_difference ∧ 
    emily_weight = 9 := by
  sorry

end emily_weight_l3038_303828


namespace system_solution_inequality_solution_set_inequality_solution_transformation_l3038_303852

-- Problem 1
theorem system_solution :
  let system (x y : ℝ) := y = x + 1 ∧ x^2 + 4*y^2 = 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ), system x₁ y₁ ∧ system x₂ y₂ ∧
    ((x₁ = 0 ∧ y₁ = 1) ∨ (x₁ = -8/5 ∧ y₁ = -3/5)) ∧
    ((x₂ = 0 ∧ y₂ = 1) ∨ (x₂ = -8/5 ∧ y₂ = -3/5)) ∧
    x₁ ≠ x₂ := by sorry

-- Problem 2
theorem inequality_solution_set (t : ℝ) :
  let solution_set := {x : ℝ | x^2 - 2*t*x + 1 > 0}
  (t < -1 ∨ t > 1 → ∃ (a b : ℝ), solution_set = {x | x < a ∨ x > b}) ∧
  (-1 < t ∧ t < 1 → solution_set = Set.univ) ∧
  (t = 1 → solution_set = {x | x ≠ 1}) ∧
  (t = -1 → solution_set = {x | x ≠ -1}) := by sorry

-- Problem 3
theorem inequality_solution_transformation (a b c : ℝ) :
  ({x : ℝ | a*x^2 + b*x + c > 0} = Set.Ioo 1 2) →
  {x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | x < -1 ∨ x > -1/2} := by sorry

end system_solution_inequality_solution_set_inequality_solution_transformation_l3038_303852


namespace no_real_solution_cubic_equation_l3038_303846

theorem no_real_solution_cubic_equation :
  ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 →
  (x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2) :=
by sorry

end no_real_solution_cubic_equation_l3038_303846


namespace complex_number_in_third_quadrant_l3038_303814

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
sorry

end complex_number_in_third_quadrant_l3038_303814


namespace right_triangle_third_side_product_l3038_303824

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((c^2 = a^2 + b^2) ∨ (b^2 = a^2 + c^2)) → 
  ((d^2 = b^2 - a^2) ∨ (b^2 = a^2 + d^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end right_triangle_third_side_product_l3038_303824


namespace student_marks_calculation_l3038_303847

theorem student_marks_calculation (total_marks : ℕ) (passing_percentage : ℚ) (failing_margin : ℕ) (student_marks : ℕ) : 
  total_marks = 500 →
  passing_percentage = 33 / 100 →
  failing_margin = 40 →
  student_marks = total_marks * passing_percentage - failing_margin →
  student_marks = 125 := by
sorry

end student_marks_calculation_l3038_303847


namespace inequality_pattern_l3038_303834

theorem inequality_pattern (x a : ℝ) : 
  x > 0 →
  x + 1/x ≥ 2 →
  x + 4/x^2 ≥ 3 →
  x + 27/x^3 ≥ 4 →
  x + a/x^4 ≥ 5 →
  a = 4^4 := by
sorry

end inequality_pattern_l3038_303834


namespace cos_value_third_quadrant_l3038_303886

theorem cos_value_third_quadrant (θ : Real) :
  tanθ = Real.sqrt 2 / 4 →
  θ > π ∧ θ < 3 * π / 2 →
  cosθ = -2 * Real.sqrt 2 / 3 := by
sorry

end cos_value_third_quadrant_l3038_303886


namespace place_value_ratio_l3038_303874

theorem place_value_ratio : 
  let number : ℝ := 37492.1053
  let ten_thousands_place_value : ℝ := 10000
  let ten_thousandths_place_value : ℝ := 0.0001
  ten_thousands_place_value / ten_thousandths_place_value = 100000000 := by
  sorry

end place_value_ratio_l3038_303874


namespace C_7_3_2_eq_10_l3038_303844

/-- A function that calculates the number of ways to select k elements from a set of n elements
    with a minimum distance of m between selected elements. -/
def C (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that C_7^(3,2) = 10 -/
theorem C_7_3_2_eq_10 : C 7 3 2 = 10 := by sorry

end C_7_3_2_eq_10_l3038_303844


namespace test_passing_requirement_l3038_303801

def total_questions : ℕ := 80
def arithmetic_questions : ℕ := 15
def algebra_questions : ℕ := 25
def geometry_questions : ℕ := 40

def arithmetic_correct_rate : ℚ := 60 / 100
def algebra_correct_rate : ℚ := 50 / 100
def geometry_correct_rate : ℚ := 70 / 100

def passing_rate : ℚ := 65 / 100

def additional_correct_answers_needed : ℕ := 3

theorem test_passing_requirement : 
  let current_correct := 
    (arithmetic_questions * arithmetic_correct_rate).floor +
    (algebra_questions * algebra_correct_rate).floor +
    (geometry_questions * geometry_correct_rate).floor
  (current_correct + additional_correct_answers_needed : ℚ) / total_questions ≥ passing_rate :=
by sorry

end test_passing_requirement_l3038_303801


namespace sin_cos_sixth_power_sum_l3038_303881

theorem sin_cos_sixth_power_sum (α : Real) (h : Real.cos (2 * α) = 1 / 5) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 25 := by
  sorry

end sin_cos_sixth_power_sum_l3038_303881


namespace five_integer_chords_l3038_303800

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 10)
  (h2 : c.distanceFromCenter = 6) : 
  countIntegerChords c = 5 :=
sorry

end five_integer_chords_l3038_303800


namespace existence_of_special_numbers_l3038_303849

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (∃ (k₁ k₂ k₃ : ℕ), 
      a * b * c = k₁ * (a + 2012) ∧
      a * b * c = k₂ * (b + 2012) ∧
      a * b * c = k₃ * (c + 2012)) :=
by sorry

end existence_of_special_numbers_l3038_303849


namespace inverse_variation_sqrt_l3038_303865

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (∃ z₀ w₀, z₀ = 8 ∧ w₀ = 4 ∧ z₀ * Real.sqrt w₀ = 8 * Real.sqrt 4) →
  (∃ w₁, 2 * Real.sqrt w₁ = 8 * Real.sqrt 4 ∧ w₁ = 64) :=
by sorry


end inverse_variation_sqrt_l3038_303865


namespace square_area_from_diagonal_l3038_303897

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  side * side = 72 := by
sorry

end square_area_from_diagonal_l3038_303897


namespace min_value_zero_l3038_303839

/-- The quadratic function for which we want to find the minimum value -/
def f (m x y : ℝ) : ℝ := 3*x^2 - 4*m*x*y + (2*m^2 + 3)*y^2 - 6*x - 9*y + 8

/-- The theorem stating the value of m that makes the minimum of f equal to 0 -/
theorem min_value_zero (m : ℝ) : 
  (∀ x y : ℝ, f m x y ≥ 0) ∧ (∃ x y : ℝ, f m x y = 0) ↔ 
  m = (6 + Real.sqrt 67.5) / 9 ∨ m = (6 - Real.sqrt 67.5) / 9 :=
sorry

end min_value_zero_l3038_303839


namespace factorization_equality_l3038_303863

theorem factorization_equality (x y : ℝ) : 2*x^2 - 8*x*y + 8*y^2 = 2*(x - 2*y)^2 := by
  sorry

end factorization_equality_l3038_303863


namespace rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l3038_303884

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (λ x y : ℝ => (x * (2 ^ (1/4 : ℝ)) + y * (5 ^ (1/2 : ℝ))) ^ 500)
  let total_terms := 501
  let is_rational_coeff := λ k : ℕ => (k % 4 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.range total_terms).filter is_rational_coeff |>.card

/-- The number of terms with rational coefficients in the expansion of (x∗∜2+y∗√5)^500 is 126 -/
theorem rational_coefficient_terms_count_is_126 : 
  rational_coefficient_terms_count = 126 := by sorry

end rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l3038_303884


namespace shaded_probability_is_half_l3038_303870

/-- Represents a game board with an equilateral triangle -/
structure GameBoard where
  /-- The number of regions the triangle is divided into -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the total number of regions is 6 -/
  total_is_six : total_regions = 6
  /-- Proof that the number of shaded regions is 3 -/
  shaded_is_three : shaded_regions = 3

/-- The probability of the spinner landing in a shaded region -/
def shaded_probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem shaded_probability_is_half (board : GameBoard) :
  shaded_probability board = 1/2 := by
  sorry

end shaded_probability_is_half_l3038_303870


namespace circle_radius_l3038_303848

theorem circle_radius (x y : Real) (h : x + y = 90 * Real.pi) :
  ∃ (r : Real), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 9 := by
sorry

end circle_radius_l3038_303848


namespace operation_result_l3038_303853

def operation (a b : ℝ) : ℝ := a * (b ^ (1/2))

theorem operation_result :
  ∀ x : ℝ, operation x 9 = 12 → x = 4 := by
sorry

end operation_result_l3038_303853


namespace greatest_x_lcm_l3038_303838

def is_lcm (a b c m : ℕ) : Prop :=
  m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧
  ∀ n : ℕ, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem greatest_x_lcm :
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end greatest_x_lcm_l3038_303838


namespace max_ab_for_line_circle_intersection_l3038_303810

/-- Given a line ax + by - 6 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 - 2x - 4y = 0
    to form a chord of length 2√5, the maximum value of ab is 9/2 -/
theorem max_ab_for_line_circle_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, a * x + b * y = 6 ∧ x^2 + y^2 - 2*x - 4*y = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 
    a * x1 + b * y1 = 6 ∧ x1^2 + y1^2 - 2*x1 - 4*y1 = 0 ∧
    a * x2 + b * y2 = 6 ∧ x2^2 + y2^2 - 2*x2 - 4*y2 = 0 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 20) →
  a * b ≤ 9/2 :=
by sorry

end max_ab_for_line_circle_intersection_l3038_303810


namespace cricket_overs_l3038_303822

theorem cricket_overs (initial_rate : ℝ) (remaining_rate : ℝ) (remaining_overs : ℝ) (target : ℝ) :
  initial_rate = 4.2 →
  remaining_rate = 8 →
  remaining_overs = 30 →
  target = 324 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * remaining_rate = target ∧ 
    initial_overs = 20 := by
  sorry

end cricket_overs_l3038_303822


namespace quadratic_equation_from_means_l3038_303823

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∃ (x : ℝ), x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l3038_303823


namespace partners_count_l3038_303804

/-- Represents the number of employees in each category -/
structure FirmComposition where
  partners : ℕ
  associates : ℕ
  managers : ℕ

/-- The initial ratio of partners : associates : managers -/
def initial_ratio : FirmComposition := ⟨2, 63, 20⟩

/-- The new ratio after hiring more employees -/
def new_ratio : FirmComposition := ⟨1, 34, 15⟩

/-- The number of additional associates hired -/
def additional_associates : ℕ := 35

/-- The number of additional managers hired -/
def additional_managers : ℕ := 10

/-- Theorem stating that the number of partners in the firm is 14 -/
theorem partners_count : ∃ (x : ℕ), 
  x * initial_ratio.partners = 14 ∧
  x * initial_ratio.associates + additional_associates = new_ratio.associates * 14 ∧
  x * initial_ratio.managers + additional_managers = new_ratio.managers * 14 :=
sorry

end partners_count_l3038_303804


namespace couscous_dishes_l3038_303875

/-- Calculates the number of dishes a restaurant can make from couscous shipments -/
theorem couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) :
  shipment1 = 7 →
  shipment2 = 13 →
  shipment3 = 45 →
  pounds_per_dish = 5 →
  (shipment1 + shipment2 + shipment3) / pounds_per_dish = 13 :=
by
  sorry

end couscous_dishes_l3038_303875


namespace parallelogram_area_is_three_l3038_303858

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (a b : Fin 2 → ℝ) : ℝ :=
  |a 0 * b 1 - a 1 * b 0|

/-- Given vectors v, w, and u, prove that the area of the parallelogram
    formed by (v + u) and w is 3 -/
theorem parallelogram_area_is_three :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![3, 1]
  let u : Fin 2 → ℝ := ![-1, 5]
  parallelogramArea (v + u) w = 3 := by
  sorry


end parallelogram_area_is_three_l3038_303858


namespace palindrome_difference_unique_l3038_303825

/-- A four-digit palindromic integer -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a d : ℕ), n = 1001 * a + 110 * d ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- A three-digit palindromic integer -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (c f : ℕ), n = 101 * c + 10 * f ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9

theorem palindrome_difference_unique :
  ∀ A B C : ℕ,
  FourDigitPalindrome A →
  FourDigitPalindrome B →
  ThreeDigitPalindrome C →
  A - B = C →
  C = 121 := by
  sorry

end palindrome_difference_unique_l3038_303825


namespace fraction_subtraction_l3038_303842

theorem fraction_subtraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  1 / x - 1 / y = 1 / 12 := by
  sorry

end fraction_subtraction_l3038_303842


namespace happy_snakes_not_purple_l3038_303882

structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

def Tom's_collection : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ (s : Snake),
  s ∈ Tom's_collection →
  (s.happy → s.can_add) ∧
  (s.purple → ¬s.can_subtract) ∧
  (¬s.can_subtract → ¬s.can_add) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end happy_snakes_not_purple_l3038_303882


namespace club_officer_selection_count_l3038_303894

/-- Represents a club with members of two genders -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of ways to select a president, vice-president, and secretary -/
def selectOfficers (club : Club) : Nat :=
  club.total_members * club.boys * (club.total_members - 2)

/-- The theorem to prove -/
theorem club_officer_selection_count (club : Club) 
  (h1 : club.total_members = 30)
  (h2 : club.boys = 15)
  (h3 : club.girls = 15)
  (h4 : club.total_members = club.boys + club.girls) :
  selectOfficers club = 12600 := by
  sorry

#eval selectOfficers { total_members := 30, boys := 15, girls := 15 }

end club_officer_selection_count_l3038_303894


namespace pipe_A_fill_time_l3038_303850

-- Define the flow rates of pipes A, B, and C
def flow_rate_A : ℝ := by sorry
def flow_rate_B : ℝ := 2 * flow_rate_A
def flow_rate_C : ℝ := 2 * flow_rate_B

-- Define the time it takes for all three pipes to fill the tank
def total_fill_time : ℝ := 4

-- Theorem stating that pipe A alone takes 28 hours to fill the tank
theorem pipe_A_fill_time :
  1 / flow_rate_A = 28 :=
by
  sorry

end pipe_A_fill_time_l3038_303850


namespace shreehari_pencils_l3038_303878

/-- Calculates the minimum number of pencils initially possessed given the number of students and pencils per student. -/
def min_initial_pencils (num_students : ℕ) (pencils_per_student : ℕ) : ℕ :=
  num_students * pencils_per_student

/-- Proves that given 25 students and 5 pencils per student, the minimum number of pencils initially possessed is 125. -/
theorem shreehari_pencils : min_initial_pencils 25 5 = 125 := by
  sorry

end shreehari_pencils_l3038_303878


namespace five_segments_create_fifteen_sections_l3038_303854

/-- The maximum number of sections created by n line segments in a rectangle,
    where each new line intersects all previously drawn lines inside the rectangle. -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => max_sections (k + 1) + k + 1

/-- The theorem stating that 5 line segments create a maximum of 15 sections. -/
theorem five_segments_create_fifteen_sections :
  max_sections 5 = 15 := by
  sorry

end five_segments_create_fifteen_sections_l3038_303854
