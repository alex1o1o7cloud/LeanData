import Mathlib

namespace cubic_expression_equals_three_l1644_164462

theorem cubic_expression_equals_three (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) 
  (h4 : p + 2*q + 3*r = 0) : (p^3 + 2*q^3 + 3*r^3) / (p*q*r) = 3 := by
  sorry

end cubic_expression_equals_three_l1644_164462


namespace initial_bees_correct_l1644_164440

/-- The initial number of bees in the colony. -/
def initial_bees : ℕ := 80000

/-- The daily loss of bees. -/
def daily_loss : ℕ := 1200

/-- The number of days after which the colony reaches a fourth of its initial size. -/
def days : ℕ := 50

/-- Theorem stating that the initial number of bees is correct given the conditions. -/
theorem initial_bees_correct : 
  initial_bees = daily_loss * days * 4 / 3 :=
by sorry

end initial_bees_correct_l1644_164440


namespace at_least_one_geq_two_l1644_164498

theorem at_least_one_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l1644_164498


namespace smallest_z_value_l1644_164486

theorem smallest_z_value (x y z : ℝ) : 
  (7 < x) → (x < 9) → (9 < y) → (y < z) → 
  (∃ (n : ℕ), y - x = n ∧ ∀ (m : ℕ), y - x ≤ m → m ≤ n) →
  (∀ (w : ℝ), (7 < w) → (w < 9) → (9 < y) → (y < z) → 
    ∃ (k : ℕ), y - w ≤ k ∧ k ≤ 7) →
  z ≥ 16 :=
by sorry

end smallest_z_value_l1644_164486


namespace negative_fraction_comparison_l1644_164484

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end negative_fraction_comparison_l1644_164484


namespace exact_blue_marbles_probability_l1644_164437

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picked : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_picks num_blue_picked) * 
  (probability_blue ^ num_blue_picked) * 
  (probability_red ^ (num_picks - num_blue_picked)) = 862 / 3417 :=
sorry

end exact_blue_marbles_probability_l1644_164437


namespace plugs_count_l1644_164452

/-- The number of pairs of mittens in the box -/
def mittens_pairs : ℕ := 150

/-- The number of pairs of plugs initially in the box -/
def initial_plugs_pairs : ℕ := mittens_pairs + 20

/-- The number of additional pairs of plugs added -/
def additional_plugs_pairs : ℕ := 30

/-- The total number of plugs after additions -/
def total_plugs : ℕ := 2 * (initial_plugs_pairs + additional_plugs_pairs)

theorem plugs_count : total_plugs = 400 := by
  sorry

end plugs_count_l1644_164452


namespace opposite_of_2023_l1644_164463

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- Theorem: The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l1644_164463


namespace unique_prime_square_l1644_164422

theorem unique_prime_square (p : ℕ) : 
  Prime p ∧ ∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2 ↔ p = 3 := by
sorry

end unique_prime_square_l1644_164422


namespace point_on_graph_l1644_164449

def is_on_graph (x y k : ℝ) : Prop := y = k * x - 2

theorem point_on_graph (k : ℝ) :
  is_on_graph 2 4 k → is_on_graph 1 1 k :=
by sorry

end point_on_graph_l1644_164449


namespace product_equals_power_of_three_l1644_164426

theorem product_equals_power_of_three : 25 * 15 * 9 * 5.4 * 3.24 = 3^10 := by
  sorry

end product_equals_power_of_three_l1644_164426


namespace probability_all_truth_l1644_164450

theorem probability_all_truth (pA pB pC pD : ℝ) 
  (hA : pA = 0.55) 
  (hB : pB = 0.60) 
  (hC : pC = 0.45) 
  (hD : pD = 0.70) : 
  pA * pB * pC * pD = 0.10395 := by
sorry

end probability_all_truth_l1644_164450


namespace perpendicular_bisector_value_l1644_164413

/-- If the line x + y = b is the perpendicular bisector of the line segment from (2,4) to (6,10), then b = 11 -/
theorem perpendicular_bisector_value (b : ℝ) : 
  (∀ (x y : ℝ), x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = (2 - 4)^2 + (4 - 7)^2 ∧ 
     (x - 4)^2 + (y - 7)^2 = (6 - 4)^2 + (10 - 7)^2)) → 
  b = 11 :=
by sorry

end perpendicular_bisector_value_l1644_164413


namespace polynomial_divisibility_l1644_164443

/-- The polynomial x^4 - 3x^3 + mx + n -/
def f (m n x : ℂ) : ℂ := x^4 - 3*x^3 + m*x + n

/-- The polynomial x^2 - 2x + 4 -/
def g (x : ℂ) : ℂ := x^2 - 2*x + 4

theorem polynomial_divisibility (m n : ℂ) :
  (∀ x, g x = 0 → f m n x = 0) →
  g (1 + Complex.I * Real.sqrt 3) = 0 →
  g (1 - Complex.I * Real.sqrt 3) = 0 →
  m = 8 ∧ n = -24 := by
  sorry

end polynomial_divisibility_l1644_164443


namespace ali_wallet_l1644_164441

def wallet_problem (num_five_dollar_bills : ℕ) (total_amount : ℕ) : ℕ := 
  let five_dollar_amount := 5 * num_five_dollar_bills
  let ten_dollar_amount := total_amount - five_dollar_amount
  ten_dollar_amount / 10

theorem ali_wallet :
  wallet_problem 7 45 = 1 := by
  sorry

end ali_wallet_l1644_164441


namespace nested_root_equation_l1644_164494

theorem nested_root_equation (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → M^(1/d + 1/(d*e) + 1/(d*e*f)) = M^(17/24)) →
  e = 4 := by
  sorry

end nested_root_equation_l1644_164494


namespace speed_increase_problem_l1644_164455

/-- The speed increase problem -/
theorem speed_increase_problem 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : initial_speed = 2) 
  (h2 : distance = 2) 
  (h3 : late_time = 1/6) 
  (h4 : early_time = 1/6) : 
  ∃ (speed_increase : ℝ), 
    speed_increase = 
      (distance / (distance / initial_speed - late_time - early_time)) - initial_speed ∧ 
    speed_increase = 1 := by
  sorry

#check speed_increase_problem

end speed_increase_problem_l1644_164455


namespace subset_P_l1644_164402

def P : Set ℝ := {x | x > -1}

theorem subset_P : {0} ⊆ P := by sorry

end subset_P_l1644_164402


namespace exp_gt_one_plus_x_l1644_164428

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end exp_gt_one_plus_x_l1644_164428


namespace even_odd_sum_difference_l1644_164464

def sum_even_2_to_40 : ℕ := (20 / 2) * (2 + 40)

def sum_odd_1_to_39 : ℕ := (20 / 2) * (1 + 39)

theorem even_odd_sum_difference : sum_even_2_to_40 - sum_odd_1_to_39 = 20 := by
  sorry

end even_odd_sum_difference_l1644_164464


namespace vehicle_Y_ahead_distance_l1644_164447

-- Define the vehicles and their properties
structure Vehicle where
  speed : ℝ
  initialPosition : ℝ

-- Define the problem parameters
def time : ℝ := 5
def vehicleX : Vehicle := { speed := 36, initialPosition := 22 }
def vehicleY : Vehicle := { speed := 45, initialPosition := 0 }

-- Define the function to calculate the position of a vehicle after a given time
def position (v : Vehicle) (t : ℝ) : ℝ :=
  v.initialPosition + v.speed * t

-- Theorem statement
theorem vehicle_Y_ahead_distance : 
  position vehicleY time - position vehicleX time = 23 := by
  sorry

end vehicle_Y_ahead_distance_l1644_164447


namespace minimum_point_of_translated_graph_l1644_164420

def f (x : ℝ) : ℝ := |x + 4| - 9

theorem minimum_point_of_translated_graph :
  ∃! (x y : ℝ), f x = y ∧ ∀ z : ℝ, f z ≥ y ∧ (x, y) = (-4, -9) := by sorry

end minimum_point_of_translated_graph_l1644_164420


namespace sixth_power_to_third_power_l1644_164417

theorem sixth_power_to_third_power (x : ℝ) (h : 728 = x^6 + 1/x^6) : 
  x^3 + 1/x^3 = Real.sqrt 730 := by
  sorry

end sixth_power_to_third_power_l1644_164417


namespace vertex_x_coordinate_is_one_l1644_164414

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem vertex_x_coordinate_is_one 
  (a b c : ℝ) 
  (h1 : quadratic a b c 0 = 3)
  (h2 : quadratic a b c 2 = 3)
  (h3 : quadratic a b c 4 = 11) :
  ∃ k : ℝ, quadratic a b c x = a * (x - 1)^2 + k := by
sorry


end vertex_x_coordinate_is_one_l1644_164414


namespace perpendicular_lines_a_values_l1644_164430

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, a^2 * x + 2 * y + 1 = 0 → x - a * y - 2 = 0 → 
   (a^2 * 1 + 2 * (-a) = 0)) → 
  (a = 2 ∨ a = 0) := by sorry

end perpendicular_lines_a_values_l1644_164430


namespace find_m_l1644_164410

/-- The value of log base 10 of 2, approximated to 4 decimal places -/
def log10_2 : ℝ := 0.3010

/-- Theorem stating that the positive integer m satisfying the given inequality is 155 -/
theorem find_m (m : ℕ) (hm_pos : m > 0) 
  (h_ineq : (10 : ℝ)^(m-1) < (2 : ℝ)^512 ∧ (2 : ℝ)^512 < (10 : ℝ)^m) : 
  m = 155 := by
  sorry

#check find_m

end find_m_l1644_164410


namespace parabola_translation_l1644_164421

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 5 0 0
  let translated := translate original 2 3
  (5 * x^2) + 3 = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c := by
  sorry

end parabola_translation_l1644_164421


namespace train_speed_calculation_train_speed_proof_l1644_164479

/-- The speed of two trains crossing each other -/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := 2 * train_length
  let relative_speed := total_distance / crossing_time
  let train_speed := relative_speed / 2
  let km_per_hour := train_speed * 3.6
  km_per_hour

/-- Proof that the speed of each train is approximately 12.01 km/hr -/
theorem train_speed_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_speed_calculation 120 36 - 12.01| < ε :=
sorry

end train_speed_calculation_train_speed_proof_l1644_164479


namespace quadrilateral_existence_l1644_164406

structure Plane where
  dummy : Unit

structure Line where
  dummy : Unit

structure Point where
  dummy : Unit

def lies_in (p : Point) (plane : Plane) : Prop := sorry

def not_in (p : Point) (plane : Plane) : Prop := sorry

def on_line (p : Point) (l : Line) : Prop := sorry

def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

def perpendicular (l1 l2 : Line) : Prop := sorry

def length_eq (s1 s2 : Point × Point) : Prop := sorry

def has_inscribed_circle (q : Point × Point × Point × Point) : Prop := sorry

theorem quadrilateral_existence 
  (P Q : Plane) (p : Line) (A C : Point) :
  intersect P Q p →
  lies_in A P →
  not_in A Q →
  lies_in C Q →
  not_in C P →
  ¬on_line A p →
  ¬on_line C p →
  ∃ (B D E : Point) (AB CD CE : Line),
    lies_in B P ∧
    lies_in D Q ∧
    parallel AB CD ∧
    parallel AB p ∧
    parallel CD p ∧
    perpendicular CE AB ∧
    length_eq (A, D) (B, C) ∧
    has_inscribed_circle (A, B, C, D) ∧
    (∃ (AE CE : Point × Point),
      (length_eq AE CE → ∃! (ABCD : Point × Point × Point × Point), ABCD = (A, B, C, D)) ∧
      (∀ x y, length_eq x y → x = AE → y = CE → 
        (∃ (ABCD1 ABCD2 : Point × Point × Point × Point), ABCD1 ≠ ABCD2 ∧ 
          (ABCD1 = (A, B, C, D) ∨ ABCD2 = (A, B, C, D))))) :=
by sorry

end quadrilateral_existence_l1644_164406


namespace five_foxes_weight_l1644_164465

/-- The weight of a single fox in kilograms. -/
def fox_weight : ℝ := sorry

/-- The weight of a single dog in kilograms. -/
def dog_weight : ℝ := fox_weight + 5

/-- The total weight of 3 foxes and 5 dogs in kilograms. -/
def total_weight : ℝ := 65

theorem five_foxes_weight :
  3 * fox_weight + 5 * dog_weight = total_weight →
  5 * fox_weight = 25 := by
  sorry

end five_foxes_weight_l1644_164465


namespace count_with_zero_up_to_3500_l1644_164475

/-- Counts the number of integers from 1 to n that contain the digit 0 in base 10 -/
def count_with_zero (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 773 numbers containing 0 up to 3500 -/
theorem count_with_zero_up_to_3500 : count_with_zero 3500 = 773 := by sorry

end count_with_zero_up_to_3500_l1644_164475


namespace unique_positive_solution_l1644_164415

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x^2 + x = 210 := by
  sorry

end unique_positive_solution_l1644_164415


namespace pat_stickers_l1644_164425

def stickers_problem (initial_stickers end_stickers : ℝ) : Prop :=
  initial_stickers - end_stickers = 22

theorem pat_stickers : stickers_problem 39 17 := by
  sorry

end pat_stickers_l1644_164425


namespace class_size_multiple_of_eight_l1644_164481

theorem class_size_multiple_of_eight (boys girls total : ℕ) : 
  girls = 7 * boys → total = boys + girls → ∃ k : ℕ, total = 8 * k := by
  sorry

end class_size_multiple_of_eight_l1644_164481


namespace f_passes_through_six_zero_f_vertex_at_four_neg_eight_l1644_164491

/-- A quadratic function passing through (6, 0) with vertex at (4, -8) -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 - 8

/-- The function f passes through the point (6, 0) -/
theorem f_passes_through_six_zero : f 6 = 0 := by sorry

/-- The vertex of f is at (4, -8) -/
theorem f_vertex_at_four_neg_eight :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a * (x - 4)^2 - 8) ∧
  (∀ (x : ℝ), f x ≥ f 4) := by sorry

end f_passes_through_six_zero_f_vertex_at_four_neg_eight_l1644_164491


namespace problem_statement_l1644_164411

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) :=
by sorry

end problem_statement_l1644_164411


namespace parabola_vertex_sum_max_l1644_164451

theorem parabola_vertex_sum_max (a U : ℤ) (h_U : U ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let N := (3 * U / 2 : ℝ) + (- 9 * a * U^2 / 4 : ℝ)
  (passes_through 0 0) ∧ 
  (passes_through (3 * U) 0) ∧ 
  (passes_through (3 * U - 1) 12) →
  N ≤ 71/4 :=
by sorry

end parabola_vertex_sum_max_l1644_164451


namespace oscar_review_questions_l1644_164470

/-- The number of questions Professor Oscar must review -/
def total_questions (num_classes : ℕ) (students_per_class : ℕ) (questions_per_exam : ℕ) : ℕ :=
  num_classes * students_per_class * questions_per_exam

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 5 35 10 = 1750 := by
  sorry

end oscar_review_questions_l1644_164470


namespace complex_product_l1644_164431

def z₁ : ℂ := 1 + 2 * Complex.I
def z₂ : ℂ := 2 - Complex.I

theorem complex_product : z₁ * z₂ = 4 + 3 * Complex.I := by
  sorry

end complex_product_l1644_164431


namespace average_sale_proof_l1644_164424

def sales_first_five : List Int := [2500, 6500, 9855, 7230, 7000]
def sales_sixth : Int := 11915
def num_months : Int := 6

theorem average_sale_proof :
  (sales_first_five.sum + sales_sixth) / num_months = 7500 := by
  sorry

end average_sale_proof_l1644_164424


namespace expression_evaluation_l1644_164427

theorem expression_evaluation : 
  |-2| + (1/3)⁻¹ - Real.sqrt 9 + (Real.sin (45 * π / 180) - 1)^0 - (-1) = 4 := by
  sorry

end expression_evaluation_l1644_164427


namespace min_height_of_box_l1644_164469

/-- Represents a rectangular box with square bases -/
structure Box where
  base : ℝ  -- side length of the square base
  height : ℝ -- height of the box
  h_positive : 0 < height
  b_positive : 0 < base

/-- The surface area of a box -/
def surface_area (box : Box) : ℝ :=
  2 * box.base^2 + 4 * box.base * box.height

/-- The constraint that the height is 5 units greater than the base -/
def height_constraint (box : Box) : Prop :=
  box.height = box.base + 5

theorem min_height_of_box (box : Box) 
  (h_constraint : height_constraint box)
  (h_surface_area : surface_area box ≥ 150) :
  box.height ≥ 10 ∧ ∃ (b : Box), height_constraint b ∧ surface_area b ≥ 150 ∧ b.height = 10 :=
sorry

end min_height_of_box_l1644_164469


namespace relationship_abc_l1644_164485

theorem relationship_abc (a b c : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ x^2 + y^2 = b ∧ x^3 + y^3 = c) → 
  a^3 - 3*a*b + 2*c = 0 := by
sorry

end relationship_abc_l1644_164485


namespace tax_rate_calculation_l1644_164416

/-- Proves that the tax rate is 30% given the specified conditions --/
theorem tax_rate_calculation (total_cost tax_free_cost : ℝ) 
  (h1 : total_cost = 20)
  (h2 : tax_free_cost = 14.7)
  (h3 : (total_cost - tax_free_cost) * 0.3 = (total_cost - tax_free_cost) * (30 / 100)) : 
  (((total_cost - tax_free_cost) * 0.3) / (total_cost - tax_free_cost)) * 100 = 30 := by
  sorry

#check tax_rate_calculation

end tax_rate_calculation_l1644_164416


namespace billys_age_l1644_164460

theorem billys_age :
  ∀ (billy_age joe_age : ℚ),
    billy_age + 5 = 2 * joe_age →
    billy_age + joe_age = 60 →
    billy_age = 115 / 3 := by
  sorry

end billys_age_l1644_164460


namespace carolyns_essay_body_sections_l1644_164477

/-- Represents the structure of Carolyn's essay -/
structure EssayStructure where
  intro_length : ℕ
  conclusion_length : ℕ
  body_section_length : ℕ
  total_length : ℕ

/-- Calculates the number of body sections in Carolyn's essay -/
def calculate_body_sections (essay : EssayStructure) : ℕ :=
  let remaining_length := essay.total_length - (essay.intro_length + essay.conclusion_length)
  remaining_length / essay.body_section_length

/-- Theorem stating that Carolyn's essay has 4 body sections -/
theorem carolyns_essay_body_sections :
  let essay := EssayStructure.mk 450 (3 * 450) 800 5000
  calculate_body_sections essay = 4 := by
  sorry

end carolyns_essay_body_sections_l1644_164477


namespace shaded_area_of_square_grid_l1644_164404

/-- The area of a square composed of 25 congruent smaller squares, 
    where the diagonal of the larger square is 10 cm, is 50 square cm. -/
theorem shaded_area_of_square_grid (d : ℝ) (n : ℕ) : 
  d = 10 → n = 25 → (d^2 / 2) * (n / n^(1/2) : ℝ)^2 = 50 := by sorry

end shaded_area_of_square_grid_l1644_164404


namespace weight_of_four_moles_of_compound_l1644_164401

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℝ) (num_moles : ℝ) : ℝ :=
  molecular_weight * num_moles

/-- Theorem: The weight of 4 moles of a compound with molecular weight 312 g/mol is 1248 grams -/
theorem weight_of_four_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 312) : weight_of_moles molecular_weight 4 = 1248 := by
  sorry

end weight_of_four_moles_of_compound_l1644_164401


namespace brick_width_is_four_l1644_164459

/-- The surface area of a rectangular prism given its length, width, and height -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular prism with length 10, height 2, and surface area 136 is 4 -/
theorem brick_width_is_four :
  ∃ w : ℝ, w > 0 ∧ surfaceArea 10 w 2 = 136 → w = 4 := by
  sorry

end brick_width_is_four_l1644_164459


namespace range_of_m_l1644_164400

def elliptical_region (x y : ℝ) : Prop := x^2 / 4 + y^2 ≤ 1

def dividing_lines (x y m : ℝ) : Prop :=
  (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x) ∨ (x = m)

def valid_coloring (n : ℕ) : Prop :=
  n = 720 ∧ ∃ (colors : Fin 6 → Type) (parts : Type) (coloring : parts → Fin 6),
    ∀ (p1 p2 : parts), p1 ≠ p2 → coloring p1 ≠ coloring p2

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x y : ℝ, elliptical_region x y → dividing_lines x y m → valid_coloring 720) ↔
    ((-2 < m ∧ m ≤ -2/3) ∨ m = 0 ∨ (2/3 ≤ m ∧ m < 2)) :=
by sorry

end range_of_m_l1644_164400


namespace probability_at_least_one_multiple_of_four_l1644_164474

def range_size : ℕ := 60
def num_selections : ℕ := 3

def multiples_of_four (n : ℕ) : ℕ := (n + 3) / 4

theorem probability_at_least_one_multiple_of_four :
  let p := 1 - (1 - multiples_of_four range_size / range_size) ^ num_selections
  p = 37 / 64 := by
  sorry

end probability_at_least_one_multiple_of_four_l1644_164474


namespace system_solution_l1644_164456

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k → 
  2*x + y = 1 → 
  x + y = 3 → 
  k = 8 := by
sorry

end system_solution_l1644_164456


namespace store_purchase_total_l1644_164492

/-- Calculate the total amount spent at the store -/
theorem store_purchase_total (initial_backpack_price initial_binder_price : ℚ)
  (backpack_increase binder_decrease : ℚ)
  (backpack_discount binder_deal sales_tax : ℚ)
  (num_binders : ℕ) :
  let new_backpack_price := initial_backpack_price + backpack_increase
  let new_binder_price := initial_binder_price - binder_decrease
  let discounted_backpack_price := new_backpack_price * (1 - backpack_discount)
  let binders_to_pay := (num_binders + 1) / 2
  let total_binder_price := new_binder_price * binders_to_pay
  let subtotal := discounted_backpack_price + total_binder_price
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_backpack_price = 50 ∧
  initial_binder_price = 20 ∧
  backpack_increase = 5 ∧
  binder_decrease = 2 ∧
  backpack_discount = 0.1 ∧
  sales_tax = 0.06 ∧
  num_binders = 3 →
  total_with_tax = 90.63 :=
by sorry

end store_purchase_total_l1644_164492


namespace complementary_angle_measure_l1644_164434

/-- Given two complementary angles A and B, where the measure of A is 3 times the measure of B,
    prove that the measure of angle A is 67.5° -/
theorem complementary_angle_measure (A B : ℝ) : 
  A + B = 90 →  -- angles A and B are complementary
  A = 3 * B →   -- measure of A is 3 times measure of B
  A = 67.5 :=   -- measure of A is 67.5°
by sorry

end complementary_angle_measure_l1644_164434


namespace specific_plot_fencing_cost_l1644_164442

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total fencing cost for a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 180 ∧
    plot.fencingCostPerMeter = 6.5 ∧
    totalFencingCost plot = 1170 := by
  sorry

end specific_plot_fencing_cost_l1644_164442


namespace area_of_ring_area_of_specific_ring_l1644_164432

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring formed by concentric circles with radii 12 and 7 is 95π -/
theorem area_of_specific_ring : 
  (π * 12^2 - π * 7^2 : ℝ) = 95 * π :=
by sorry

end area_of_ring_area_of_specific_ring_l1644_164432


namespace simple_interest_difference_l1644_164446

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal rate time : ℕ) 
  (h_principal : principal = 2800)
  (h_rate : rate = 4)
  (h_time : time = 5) : 
  principal - (principal * rate * time) / 100 = 2240 := by
  sorry

end simple_interest_difference_l1644_164446


namespace sum_of_nested_logs_l1644_164482

-- Define the logarithm functions
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sum_of_nested_logs (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧
  log 3 (log 4 (log 2 y)) = 0 ∧
  log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 := by
  sorry

end sum_of_nested_logs_l1644_164482


namespace correct_regression_equation_l1644_164418

/-- Represents the selling price of a product in yuan per piece -/
def SellingPrice : Type := ℝ

/-- Represents the sales volume of a product in pieces -/
def SalesVolume : Type := ℝ

/-- Represents a regression equation for sales volume based on selling price -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Indicates that two variables are negatively correlated -/
def NegativelyCorrelated (x : Type) (y : Type) : Prop := sorry

/-- Checks if a regression equation is valid for negatively correlated variables -/
def IsValidRegression (eq : RegressionEquation) (x : Type) (y : Type) : Prop := 
  NegativelyCorrelated x y → eq.slope < 0

/-- The correct regression equation for the given problem -/
def CorrectEquation : RegressionEquation := { slope := -2, intercept := 100 }

/-- Theorem stating that the CorrectEquation is valid for the given problem -/
theorem correct_regression_equation : 
  IsValidRegression CorrectEquation SellingPrice SalesVolume := sorry

end correct_regression_equation_l1644_164418


namespace tomatoes_picked_today_l1644_164488

/-- Represents the number of tomatoes in various states --/
structure TomatoCount where
  initial : ℕ
  pickedYesterday : ℕ
  leftAfterYesterday : ℕ

/-- Theorem: The number of tomatoes picked today is equal to the initial number
    minus the number left after yesterday's picking --/
theorem tomatoes_picked_today (t : TomatoCount)
  (h1 : t.initial = 160)
  (h2 : t.pickedYesterday = 56)
  (h3 : t.leftAfterYesterday = 104)
  : t.initial - t.leftAfterYesterday = 56 := by
  sorry


end tomatoes_picked_today_l1644_164488


namespace hyperbola_asymptote_slopes_l1644_164403

/-- The slopes of the asymptotes for the hyperbola described by the equation x²/144 - y²/81 = 1 are ±3/4 -/
theorem hyperbola_asymptote_slopes (x y : ℝ) :
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ (m : ℝ), m = 3/4 ∧ (∀ (x' y' : ℝ), x'^2 / 144 - y'^2 / 81 = 0 → y' = m * x' ∨ y' = -m * x') :=
by sorry

end hyperbola_asymptote_slopes_l1644_164403


namespace bus_time_calculation_l1644_164409

def wake_up_time : ℕ := 6 * 60 + 45
def bus_departure_time : ℕ := 7 * 60 + 15
def class_duration : ℕ := 45
def num_classes : ℕ := 7
def lunch_duration : ℕ := 20
def science_lab_duration : ℕ := 60
def additional_time : ℕ := 90
def arrival_time : ℕ := 15 * 60 + 50

def total_school_time : ℕ := 
  num_classes * class_duration + lunch_duration + science_lab_duration + additional_time

def total_away_time : ℕ := arrival_time - bus_departure_time

theorem bus_time_calculation : 
  total_away_time - total_school_time = 30 := by sorry

end bus_time_calculation_l1644_164409


namespace greatest_four_digit_multiple_of_17_l1644_164429

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 9996 ∧ 
  n % 17 = 0 ∧ 
  n ≤ 9999 ∧ 
  ∀ m : ℕ, m % 17 = 0 ∧ m ≤ 9999 → m ≤ n := by
sorry

end greatest_four_digit_multiple_of_17_l1644_164429


namespace abs_three_minus_pi_l1644_164483

theorem abs_three_minus_pi : |3 - Real.pi| = Real.pi - 3 := by
  sorry

end abs_three_minus_pi_l1644_164483


namespace garden_cut_percentage_l1644_164435

theorem garden_cut_percentage (rows : ℕ) (flowers_per_row : ℕ) (remaining : ℕ) :
  rows = 50 →
  flowers_per_row = 400 →
  remaining = 8000 →
  (rows * flowers_per_row - remaining : ℚ) / (rows * flowers_per_row) * 100 = 60 := by
  sorry

end garden_cut_percentage_l1644_164435


namespace sin_alpha_for_point_3_4_l1644_164458

/-- Given an angle α where a point on its terminal side has coordinates (3,4), prove that sin α = 4/5 -/
theorem sin_alpha_for_point_3_4 (α : Real) :
  (∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 4/5 := by
sorry

end sin_alpha_for_point_3_4_l1644_164458


namespace intersection_A_complement_B_l1644_164467

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of B
def complement_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 < x ∧ x < 4} := by
  sorry

end intersection_A_complement_B_l1644_164467


namespace trig_expression_equality_l1644_164489

theorem trig_expression_equality : 
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) - 
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) = 
  Real.sqrt 3 / 2 := by sorry

end trig_expression_equality_l1644_164489


namespace city_population_dynamics_l1644_164496

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birthRate : ℝ  -- Average birth rate per second
  netIncrease : ℝ  -- Net population increase per second
  deathRate : ℝ  -- Average death rate per second

/-- Theorem stating the relationship between birth rate, net increase, and death rate --/
theorem city_population_dynamics (city : CityPopulation) 
  (h1 : city.birthRate = 3.5)
  (h2 : city.netIncrease = 2) :
  city.deathRate = 1.5 := by
  sorry

#check city_population_dynamics

end city_population_dynamics_l1644_164496


namespace cost_price_calculation_l1644_164454

/-- Proves that the cost price of an article is 95 given the specified conditions -/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  marked_price = 125 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ), 
    cost_price = 95 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end cost_price_calculation_l1644_164454


namespace elementary_classes_count_l1644_164444

/-- The number of elementary school classes in each school -/
def elementary_classes : ℕ := 4

/-- The number of schools -/
def num_schools : ℕ := 2

/-- The number of middle school classes in each school -/
def middle_classes : ℕ := 5

/-- The number of soccer balls donated per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

theorem elementary_classes_count :
  elementary_classes * num_schools * balls_per_class +
  middle_classes * num_schools * balls_per_class = total_balls :=
by sorry

end elementary_classes_count_l1644_164444


namespace factorial_20_19_div_5_is_perfect_square_l1644_164478

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_20_19_div_5_is_perfect_square :
  is_perfect_square ((factorial 20 * factorial 19) / 5) := by
  sorry

end factorial_20_19_div_5_is_perfect_square_l1644_164478


namespace solution_set_f_gt_2x_plus_1_range_of_t_for_f_geq_g_l1644_164407

-- Define the functions f and g
def f (x : ℝ) := |x - 1|
def g (t x : ℝ) := t * |x| - 2

-- Statement 1: Solution set of f(x) > 2x+1
theorem solution_set_f_gt_2x_plus_1 :
  {x : ℝ | f x > 2 * x + 1} = {x : ℝ | x < 0} := by sorry

-- Statement 2: Range of t for which f(x) ≥ g(x) holds for all x ∈ ℝ
theorem range_of_t_for_f_geq_g :
  ∀ t : ℝ, (∀ x : ℝ, f x ≥ g t x) ↔ t ≤ 1 := by sorry

end solution_set_f_gt_2x_plus_1_range_of_t_for_f_geq_g_l1644_164407


namespace unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l1644_164436

theorem unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2 :
  ∃! n : ℕ+, 18 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.2 :=
by sorry

end unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l1644_164436


namespace gdp_scientific_notation_l1644_164412

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original GDP value in trillions of dollars -/
def originalGDP : ℝ := 1.337

/-- The number of significant figures to use -/
def sigFigs : ℕ := 3

theorem gdp_scientific_notation :
  toScientificNotation (originalGDP * 1000000000000) sigFigs =
    ScientificNotation.mk 1.34 12 (by norm_num) :=
  sorry

end gdp_scientific_notation_l1644_164412


namespace least_number_for_divisibility_l1644_164439

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28457 + y) % 37 = 0 ∧ (28457 + y) % 59 = 0 ∧ (28457 + y) % 67 = 0)) ∧
  (28457 + x) % 37 = 0 ∧ (28457 + x) % 59 = 0 ∧ (28457 + x) % 67 = 0 →
  x = 117804 :=
by sorry

end least_number_for_divisibility_l1644_164439


namespace arctan_equation_solution_l1644_164473

theorem arctan_equation_solution (y : ℝ) :
  4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = 1251 := by
  sorry

end arctan_equation_solution_l1644_164473


namespace class_size_from_average_change_l1644_164493

theorem class_size_from_average_change 
  (original_mark : ℕ) 
  (incorrect_mark : ℕ)
  (mark_difference : ℕ)
  (average_increase : ℚ) :
  incorrect_mark = original_mark + mark_difference →
  mark_difference = 20 →
  average_increase = 1/2 →
  (mark_difference : ℚ) / (class_size : ℕ) = average_increase →
  class_size = 40 := by
sorry

end class_size_from_average_change_l1644_164493


namespace inequality_equivalence_l1644_164468

theorem inequality_equivalence (x : ℝ) : 
  (1/2: ℝ) ^ (x^2 - 2*x + 3) < (1/2 : ℝ) ^ (2*x^2 + 3*x - 3) ↔ -6 < x ∧ x < 1 := by
  sorry

end inequality_equivalence_l1644_164468


namespace average_problem_l1644_164405

theorem average_problem (t b c : ℝ) (h : (t + b + c + 29) / 4 = 15) :
  (t + b + c + 14 + 15) / 5 = 12 := by
  sorry

end average_problem_l1644_164405


namespace count_large_glasses_l1644_164433

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_jelly_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_jelly_beans : ℕ := 25

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jelly beans used to fill all glasses -/
def total_jelly_beans : ℕ := 325

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

theorem count_large_glasses : 
  large_glass_jelly_beans * num_large_glasses + 
  small_glass_jelly_beans * num_small_glasses = total_jelly_beans :=
by sorry

end count_large_glasses_l1644_164433


namespace constant_sum_and_square_sum_implies_constant_S_l1644_164490

theorem constant_sum_and_square_sum_implies_constant_S 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) : 
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * (a^2 + b^2 + c^2 + d^2) = 7.5 := by
sorry

end constant_sum_and_square_sum_implies_constant_S_l1644_164490


namespace modular_inverse_35_mod_36_l1644_164495

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_35_mod_36_l1644_164495


namespace slope_range_l1644_164471

theorem slope_range (a : ℝ) :
  let k := -(1 / (a^2 + Real.sqrt 3))
  5 * Real.pi / 6 ≤ Real.arctan k ∧ Real.arctan k < Real.pi :=
by sorry

end slope_range_l1644_164471


namespace john_average_score_l1644_164499

def john_scores : List ℝ := [88, 95, 90, 84, 91]

theorem john_average_score : (john_scores.sum / john_scores.length) = 89.6 := by
  sorry

end john_average_score_l1644_164499


namespace water_height_in_conical_tank_l1644_164472

/-- The height of water in an inverted conical tank -/
theorem water_height_in_conical_tank 
  (tank_radius : ℝ) 
  (tank_height : ℝ) 
  (water_volume_percentage : ℝ) 
  (h : water_volume_percentage = 0.4) 
  (r : tank_radius = 10) 
  (h : tank_height = 60) : 
  ∃ (water_height : ℝ), water_height = 12 * (3 ^ (1/3 : ℝ)) :=
sorry

end water_height_in_conical_tank_l1644_164472


namespace rectangle_ratio_l1644_164457

/-- Given a configuration of four congruent rectangles arranged around an inner square,
    where the area of the outer square is 9 times the area of the inner square,
    the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) (h5 : x + s = 3*s) : x / y = 2 := by
  sorry

end rectangle_ratio_l1644_164457


namespace center_value_of_arithmetic_array_l1644_164408

/-- Represents a 3x3 array with arithmetic sequences in rows and columns -/
def ArithmeticArray := Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if a sequence of three real numbers is arithmetic -/
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

/-- Properties of the arithmetic array -/
def arithmetic_array_properties (A : ArithmeticArray) : Prop :=
  ∀ i : Fin 3,
    (is_arithmetic_sequence (A i 0) (A i 1) (A i 2)) ∧
    (is_arithmetic_sequence (A 0 i) (A 1 i) (A 2 i))

theorem center_value_of_arithmetic_array (A : ArithmeticArray) 
  (h_props : arithmetic_array_properties A)
  (h_first_row : A 0 0 = 3 ∧ A 0 2 = 15)
  (h_last_row : A 2 0 = 9 ∧ A 2 2 = 33) :
  A 1 1 = 15 := by
  sorry

end center_value_of_arithmetic_array_l1644_164408


namespace quilt_transformation_l1644_164438

theorem quilt_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
  sorry

end quilt_transformation_l1644_164438


namespace profit_percentage_per_item_l1644_164461

theorem profit_percentage_per_item (total_cost : ℝ) (num_bought num_sold : ℕ) 
  (h1 : num_bought = 30)
  (h2 : num_sold = 20)
  (h3 : total_cost > 0)
  (h4 : num_bought > num_sold)
  (h5 : num_sold * (total_cost / num_bought) = total_cost) :
  (((total_cost / num_sold) - (total_cost / num_bought)) / (total_cost / num_bought)) * 100 = 50 := by
  sorry

end profit_percentage_per_item_l1644_164461


namespace systematic_sampling_identification_l1644_164480

/-- A sampling method is a function that selects elements from a population. -/
def SamplingMethod := Type → Type

/-- Systematic sampling is a method where samples are selected at regular intervals. -/
def IsSystematicSampling (m : SamplingMethod) : Prop := sorry

/-- Method 1: Sampling from numbered balls with a fixed interval. -/
def Method1 : SamplingMethod := sorry

/-- Method 2: Sampling products from a conveyor belt at fixed time intervals. -/
def Method2 : SamplingMethod := sorry

/-- Method 3: Random sampling at a shopping mall entrance. -/
def Method3 : SamplingMethod := sorry

/-- Method 4: Sampling moviegoers in specific seats. -/
def Method4 : SamplingMethod := sorry

/-- Theorem stating which methods are systematic sampling. -/
theorem systematic_sampling_identification :
  IsSystematicSampling Method1 ∧
  IsSystematicSampling Method2 ∧
  ¬IsSystematicSampling Method3 ∧
  IsSystematicSampling Method4 := by sorry

end systematic_sampling_identification_l1644_164480


namespace parallelogram_perimeter_area_sum_l1644_164453

-- Define a parallelogram type
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

-- Define the property of having right or obtuse angles
def has_right_or_obtuse_angles (p : Parallelogram) : Prop :=
  sorry

-- Define the perimeter of the parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  sorry

-- Define the area of the parallelogram
def area (p : Parallelogram) : ℝ :=
  sorry

-- Theorem statement
theorem parallelogram_perimeter_area_sum :
  ∀ p : Parallelogram,
  p.v1 = (6, 3) ∧ p.v2 = (9, 7) ∧ p.v3 = (2, 0) ∧
  has_right_or_obtuse_angles p →
  perimeter p + area p = 48 :=
sorry

end parallelogram_perimeter_area_sum_l1644_164453


namespace mike_work_hours_l1644_164476

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def paint_time : ℕ := 45
def engine_service_time : ℕ := 60

def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2
def cars_painted : ℕ := 4
def engines_serviced : ℕ := 3

def total_minutes : ℕ := 
  wash_time * cars_washed + 
  oil_change_time * cars_oil_changed + 
  tire_change_time * tire_sets_changed + 
  paint_time * cars_painted + 
  engine_service_time * engines_serviced

theorem mike_work_hours : total_minutes / 60 = 10 := by
  sorry

end mike_work_hours_l1644_164476


namespace cylinder_radius_approximation_l1644_164423

noncomputable def cylinder_radius (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  let cylinder_volume := 2 * rectangle_area
  let cylinder_height := square_side
  Real.sqrt (cylinder_volume / (Real.pi * cylinder_height))

theorem cylinder_radius_approximation :
  ∀ (ε : ℝ), ε > 0 →
  abs (cylinder_radius 2500 10 - 1.59514) < ε :=
sorry

end cylinder_radius_approximation_l1644_164423


namespace volume_formula_l1644_164487

/-- A pyramid with a rectangular base -/
structure Pyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side AD of the base -/
  ad : ℝ
  /-- Angle AQB where Q is the apex -/
  θ : ℝ
  /-- Assertion that AB = 2 -/
  hab : ab = 2
  /-- Assertion that AD = 1 -/
  had : ad = 1
  /-- Assertion that Q is directly above the center of the base -/
  hcenter : True
  /-- Assertion that Q is equidistant from all vertices -/
  hequidistant : True

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4)

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula (p : Pyramid) : volume p = (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4) := by
  sorry

end volume_formula_l1644_164487


namespace female_population_l1644_164445

theorem female_population (total_population : ℕ) (male_ratio female_ratio : ℕ) 
  (h_total : total_population = 500)
  (h_ratio : male_ratio = 3 ∧ female_ratio = 2) : 
  (female_ratio * total_population) / (male_ratio + female_ratio) = 200 := by
  sorry

end female_population_l1644_164445


namespace problem_solution_l1644_164466

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
  sorry

end problem_solution_l1644_164466


namespace train_speed_calculation_l1644_164448

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed. -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 275 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1644_164448


namespace geometric_sequence_312th_term_l1644_164497

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a₁ : ℝ  -- first term
  r : ℝ   -- common ratio
  
/-- The nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a₁ * seq.r ^ (n - 1)

/-- Theorem: The 312th term of the specific geometric sequence -/
theorem geometric_sequence_312th_term :
  let seq : GeometricSequence := { a₁ := 12, r := -1/2 }
  seq.nthTerm 312 = -12 * (1/2)^311 := by
  sorry


end geometric_sequence_312th_term_l1644_164497


namespace equation_real_roots_range_l1644_164419

theorem equation_real_roots_range (a : ℝ) : 
  (∀ x : ℝ, (2 + 3*a) / (5 - a) > 0) ↔ a ∈ Set.Ioo (-2/3 : ℝ) 5 := by sorry

end equation_real_roots_range_l1644_164419
