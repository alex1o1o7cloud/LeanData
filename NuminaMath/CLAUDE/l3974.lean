import Mathlib

namespace jims_pantry_flour_l3974_397444

/-- The amount of flour Jim has in the pantry -/
def flour_in_pantry : ℕ := sorry

/-- The total amount of flour Jim has in the cupboard and on the kitchen counter -/
def flour_elsewhere : ℕ := 300

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf : ℕ := 200

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := 2

theorem jims_pantry_flour :
  flour_in_pantry = 100 :=
by sorry

end jims_pantry_flour_l3974_397444


namespace probability_failed_math_given_failed_chinese_l3974_397432

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16) 
  (h2 : failed_chinese = 0.07) 
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 :=
by sorry

end probability_failed_math_given_failed_chinese_l3974_397432


namespace linear_equation_m_value_l3974_397417

theorem linear_equation_m_value : 
  ∃! m : ℤ, (abs m - 4 = 1) ∧ (m - 5 ≠ 0) :=
by sorry

end linear_equation_m_value_l3974_397417


namespace expression_value_l3974_397485

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end expression_value_l3974_397485


namespace expression_equality_l3974_397488

theorem expression_equality (w : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.00 / Real.sqrt w = 2.650793650793651) → 
  w = 0.49 := by
  sorry

end expression_equality_l3974_397488


namespace function_machine_output_l3974_397468

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then
    step1 - 8
  else
    step1 + 10

theorem function_machine_output :
  function_machine 7 = 13 := by
  sorry

end function_machine_output_l3974_397468


namespace pyramid_volume_in_unit_cube_l3974_397427

/-- The volume of a pyramid within a unit cube, where the pyramid's vertex is at one corner of the cube
    and its base is a triangle formed by the midpoints of three adjacent edges meeting at the opposite corner -/
theorem pyramid_volume_in_unit_cube : ∃ V : ℝ, V = Real.sqrt 3 / 24 :=
  sorry

end pyramid_volume_in_unit_cube_l3974_397427


namespace sum_maximized_at_14_l3974_397433

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (83 - 3 * n) / 2

/-- The theorem stating that the sum is maximized when n = 14 -/
theorem sum_maximized_at_14 :
  ∀ k : ℕ, k ≠ 0 → S 14 ≥ S k :=
sorry

end sum_maximized_at_14_l3974_397433


namespace diamond_four_three_l3974_397479

/-- Diamond operation: a ◇ b = 4a + 3b - ab + a² + b² -/
def diamond (a b : ℝ) : ℝ := 4*a + 3*b - a*b + a^2 + b^2

theorem diamond_four_three : diamond 4 3 = 38 := by sorry

end diamond_four_three_l3974_397479


namespace ellipse_equation_l3974_397408

/-- Given an ellipse with foci on the y-axis, sum of distances from any point to the foci equal to 8,
    and focal length 2√15, prove that its standard equation is (y²/16) + x² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : 2 * c = 2 * Real.sqrt 15)
    (h3 : a ^ 2 = b ^ 2 + c ^ 2) :
  ∀ (x y : ℝ), (y ^ 2 / 16 + x ^ 2 = 1) ↔ (y ^ 2 / a ^ 2 + x ^ 2 / b ^ 2 = 1) :=
by sorry

end ellipse_equation_l3974_397408


namespace tanya_erasers_l3974_397446

/-- Given the number of erasers for Hanna, Rachel, and Tanya, prove that Tanya has 20 erasers -/
theorem tanya_erasers (h r t : ℕ) (tr : ℕ) : 
  h = 2 * r →  -- Hanna has twice as many erasers as Rachel
  r = tr / 2 - 3 →  -- Rachel has three less than one-half as many erasers as Tanya has red erasers
  tr = t / 2 →  -- Half of Tanya's erasers are red
  h = 4 →  -- Hanna has 4 erasers
  t = 20 := by sorry

end tanya_erasers_l3974_397446


namespace enlarged_lawn_area_l3974_397445

theorem enlarged_lawn_area (initial_width : ℝ) (initial_area : ℝ) (new_width : ℝ) :
  initial_width = 8 →
  initial_area = 640 →
  new_width = 16 →
  let length : ℝ := initial_area / initial_width
  let new_area : ℝ := length * new_width
  new_area = 1280 := by
  sorry

end enlarged_lawn_area_l3974_397445


namespace quadratic_root_implies_m_l3974_397472

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 3 * x^2 - x + m = 0) → m = -2 := by
  sorry

end quadratic_root_implies_m_l3974_397472


namespace pickled_vegetables_grade_C_l3974_397404

/-- Represents the number of boxes of pickled vegetables in each grade -/
structure GradeBoxes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- 
Given:
- There are 420 boxes of pickled vegetables in total
- The vegetables are classified into three grades: A, B, and C
- m, n, and t are the number of boxes sampled from grades A, B, and C, respectively
- 2t = m + n

Prove that the number of boxes classified as grade C is 140
-/
theorem pickled_vegetables_grade_C (boxes : GradeBoxes) 
  (total_boxes : boxes.A + boxes.B + boxes.C = 420)
  (sample_relation : ∃ (m n t : ℕ), 2 * t = m + n) :
  boxes.C = 140 := by
  sorry

end pickled_vegetables_grade_C_l3974_397404


namespace nature_of_c_l3974_397497

theorem nature_of_c (a c : ℝ) (h : (2*a - 1) / (-3) < -(c + 1) / (-4)) :
  (c < 0 ∨ c > 0) ∧ c ≠ -1 :=
sorry

end nature_of_c_l3974_397497


namespace part_one_part_two_l3974_397495

-- Part 1
theorem part_one (x y : ℝ) : 
  y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 3 →
  x = 2 →
  x - y = -1 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  x = Real.sqrt 2 →
  (x / (x - 2)) / (2 + x - 4 / (2 - x)) = Real.sqrt 2 / 2 := by sorry

end part_one_part_two_l3974_397495


namespace exam_results_l3974_397461

theorem exam_results (total_students : ℕ) (second_division_percent : ℚ) 
  (just_passed : ℕ) (h1 : total_students = 300) 
  (h2 : second_division_percent = 54/100) (h3 : just_passed = 57) : 
  (1 - second_division_percent - (just_passed : ℚ) / total_students) = 27/100 := by
  sorry

end exam_results_l3974_397461


namespace inequality_proof_l3974_397423

theorem inequality_proof (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 := by
sorry

end inequality_proof_l3974_397423


namespace smallest_appended_digits_for_divisibility_l3974_397419

theorem smallest_appended_digits_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, 
    (n = 2014) ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0) ∧
    (∀ j : ℕ, j < 10000 → 
      (∃ m : ℕ, m < 10 ∧ (n * j + k) % m ≠ 0))) → 
  (∃ k : ℕ, k < 10000 ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0)) :=
sorry

end smallest_appended_digits_for_divisibility_l3974_397419


namespace same_price_at_12_sheets_unique_equal_price_point_l3974_397453

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  per_sheet : ℚ
  sitting_fee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.per_sheet * sheets + company.sitting_fee

/-- John's Photo World pricing -/
def johns_photo_world : PhotoCompany :=
  { per_sheet := 2.75, sitting_fee := 125 }

/-- Sam's Picture Emporium pricing -/
def sams_picture_emporium : PhotoCompany :=
  { per_sheet := 1.50, sitting_fee := 140 }

/-- Theorem stating that the two companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  total_cost johns_photo_world 12 = total_cost sams_picture_emporium 12 :=
by sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_point (sheets : ℚ) :
  total_cost johns_photo_world sheets = total_cost sams_picture_emporium sheets ↔ sheets = 12 :=
by sorry

end same_price_at_12_sheets_unique_equal_price_point_l3974_397453


namespace x_range_for_inequality_l3974_397441

theorem x_range_for_inequality (x : ℝ) : 
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) →
  ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) :=
by sorry

end x_range_for_inequality_l3974_397441


namespace sum_of_qp_values_l3974_397420

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end sum_of_qp_values_l3974_397420


namespace quadratic_solution_sum_l3974_397475

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → x * y = 7/5 →
  ∃ (a b c d : ℝ), x = (a + b * Real.sqrt c) / d ∧ 
                    x = (a - b * Real.sqrt c) / d ∧
                    a + b + c + d = 521 := by
  sorry

end quadratic_solution_sum_l3974_397475


namespace divisibility_by_53_l3974_397435

theorem divisibility_by_53 (n : ℕ) : 53 ∣ (10^(n+3) + 17) := by sorry

end divisibility_by_53_l3974_397435


namespace sandy_younger_than_molly_l3974_397414

theorem sandy_younger_than_molly (sandy_age molly_age : ℕ) : 
  sandy_age = 63 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age - sandy_age = 18 := by
sorry

end sandy_younger_than_molly_l3974_397414


namespace total_rent_is_105_l3974_397434

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  share : ℚ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def calculate_total_rent (a b c : RentShare) : ℚ :=
  let total_oxen_months : ℕ := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let rent_per_oxen_month : ℚ := c.share / (c.oxen * c.months)
  rent_per_oxen_month * total_oxen_months

/-- Theorem: The total rent of the pasture is 105.00 given the specified conditions -/
theorem total_rent_is_105 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.share = 26.999999999999996) :
  calculate_total_rent a b c = 105 :=
by sorry

end total_rent_is_105_l3974_397434


namespace fgh_supermarket_difference_l3974_397425

/-- Prove that the difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference : ∀ (total us : ℕ),
  total = 84 →
  us = 49 →
  us > total - us →
  us - (total - us) = 14 := by
  sorry

end fgh_supermarket_difference_l3974_397425


namespace colin_speed_l3974_397493

/-- Given the relationships between the speeds of Colin, Brandon, Tony, Bruce, and Daniel,
    prove that Colin's speed is 8 miles per hour when Bruce's speed is 1 mile per hour. -/
theorem colin_speed (bruce tony brandon colin daniel : ℝ) : 
  bruce = 1 →
  tony = 2 * bruce →
  brandon = (1/3) * tony^2 →
  colin = 6 * brandon →
  daniel = (1/4) * colin →
  colin = 8 := by
sorry

end colin_speed_l3974_397493


namespace floor_width_is_eight_meters_l3974_397443

/-- Proves that a rectangular floor with given dimensions has a width of 8 meters -/
theorem floor_width_is_eight_meters
  (floor_length : ℝ)
  (rug_area : ℝ)
  (strip_width : ℝ)
  (h1 : floor_length = 10)
  (h2 : rug_area = 24)
  (h3 : strip_width = 2)
  : ∃ (floor_width : ℝ),
    floor_width = 8 ∧
    rug_area = (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) :=
by
  sorry


end floor_width_is_eight_meters_l3974_397443


namespace answer_key_combinations_l3974_397415

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 5

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem: The number of ways to create an answer key for the quiz is 480 -/
theorem answer_key_combinations : 
  validTrueFalseCombinations * multipleChoiceCombinations = 480 := by
  sorry

end answer_key_combinations_l3974_397415


namespace hasans_plates_l3974_397466

/-- Proves the number of plates initially in Hasan's box -/
theorem hasans_plates
  (plate_weight : ℕ)
  (max_weight_oz : ℕ)
  (removed_plates : ℕ)
  (h1 : plate_weight = 10)
  (h2 : max_weight_oz = 20 * 16)
  (h3 : removed_plates = 6) :
  (max_weight_oz + removed_plates * plate_weight) / plate_weight = 38 := by
  sorry

end hasans_plates_l3974_397466


namespace percentage_not_covering_politics_l3974_397449

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 10

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Represents the total number of reporters (assumed for calculation purposes) -/
def total_reporters : ℝ := 100

/-- Theorem stating that 86% of reporters do not cover politics -/
theorem percentage_not_covering_politics :
  (total_reporters - (local_politics_coverage / (100 - non_local_political_coverage) * 100)) / total_reporters * 100 = 86 := by
  sorry

end percentage_not_covering_politics_l3974_397449


namespace sum_of_real_and_imag_parts_of_z_l3974_397407

theorem sum_of_real_and_imag_parts_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := (1 + 2*i) / i
  (z.re + z.im : ℝ) = 1 := by
sorry

end sum_of_real_and_imag_parts_of_z_l3974_397407


namespace no_common_elements_l3974_397437

-- Define the sequence Pn(x)
def P : ℕ → (ℝ → ℝ)
  | 0 => λ x => x
  | 1 => λ x => 4 * x^3 + 3 * x
  | (n + 2) => λ x => (4 * x^2 + 2) * P (n + 1) x - P n x

-- Define the set A(m)
def A (m : ℝ) : Set ℝ := {y | ∃ n : ℕ, y = P n m}

-- Theorem statement
theorem no_common_elements (m : ℝ) (h : m > 0) :
  ∀ n k : ℕ, P n m ≠ P k (m + 4) :=
by sorry

end no_common_elements_l3974_397437


namespace smallest_n_value_l3974_397474

theorem smallest_n_value : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), m < n → ¬(∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*m ∧ 24*m = 30*22)) ∧ 
  (∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*n ∧ 24*n = 30*22) ∧
  n = 30 := by
  sorry

end smallest_n_value_l3974_397474


namespace angle_identities_l3974_397486

theorem angle_identities (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 4) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 12) = (Real.sqrt 6 + 3) / 6 ∧ 
  Real.cos (2 * α + π / 6) = (2 * Real.sqrt 6 - 1) / 6 := by
sorry

end angle_identities_l3974_397486


namespace quarters_remaining_l3974_397458

theorem quarters_remaining (initial_quarters : ℕ) (payment_dollars : ℕ) (quarters_per_dollar : ℕ) : 
  initial_quarters = 160 →
  payment_dollars = 35 →
  quarters_per_dollar = 4 →
  initial_quarters - (payment_dollars * quarters_per_dollar) = 20 :=
by sorry

end quarters_remaining_l3974_397458


namespace consecutive_integers_problem_l3974_397464

theorem consecutive_integers_problem (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end consecutive_integers_problem_l3974_397464


namespace distance_ratio_proof_l3974_397428

/-- Proves that the ratio of distances covered at different speeds is 1:1 given specific conditions -/
theorem distance_ratio_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3600)
  (h2 : speed1 = 90)
  (h3 : speed2 = 180)
  (h4 : total_time = 30)
  (h5 : ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time) :
  ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time ∧ d1 = d2 := by
  sorry

#check distance_ratio_proof

end distance_ratio_proof_l3974_397428


namespace package_servings_l3974_397489

/-- The number of servings in a package of candy. -/
def servings_in_package (calories_per_serving : ℕ) (calories_in_half : ℕ) : ℕ :=
  (2 * calories_in_half) / calories_per_serving

/-- Theorem: Given a package where each serving has 120 calories and half the package contains 180 calories, 
    prove that there are 3 servings in the package. -/
theorem package_servings : servings_in_package 120 180 = 3 := by
  sorry

end package_servings_l3974_397489


namespace oreo_cheesecake_solution_l3974_397470

def oreo_cheesecake_problem (graham_boxes_bought : ℕ) (oreo_packets_bought : ℕ) 
  (graham_boxes_per_cake : ℕ) (graham_boxes_leftover : ℕ) : ℕ :=
  let cakes_made := (graham_boxes_bought - graham_boxes_leftover) / graham_boxes_per_cake
  oreo_packets_bought / cakes_made

theorem oreo_cheesecake_solution :
  oreo_cheesecake_problem 14 15 2 4 = 3 := by
  sorry

end oreo_cheesecake_solution_l3974_397470


namespace youngest_son_park_visits_l3974_397405

theorem youngest_son_park_visits (season_pass_cost : ℝ) (oldest_son_visits : ℕ) (youngest_son_cost_per_trip : ℝ) :
  season_pass_cost = 100 →
  oldest_son_visits = 35 →
  youngest_son_cost_per_trip = 4 →
  ∃ (youngest_son_visits : ℕ), 
    (season_pass_cost / youngest_son_visits) = youngest_son_cost_per_trip ∧
    youngest_son_visits = 25 :=
by sorry

end youngest_son_park_visits_l3974_397405


namespace product_of_D_coordinates_l3974_397402

-- Define the points
def C : ℝ × ℝ := (-2, -7)
def M : ℝ × ℝ := (4, -3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- State the theorem
theorem product_of_D_coordinates : 
  (M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) → D.1 * D.2 = 10 := by
  sorry

end product_of_D_coordinates_l3974_397402


namespace inequality_proof_l3974_397477

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x) / (1 + x^2) + (3*y^2 - y) / (1 + y^2) + (3*z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end inequality_proof_l3974_397477


namespace rectangle_area_12_l3974_397494

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

def valid_rectangles : Set (ℕ+ × ℕ+) :=
  {p | rectangle_area p.1 p.2 = 12}

theorem rectangle_area_12 :
  valid_rectangles = {(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)} := by
  sorry

end rectangle_area_12_l3974_397494


namespace min_steps_ladder_l3974_397430

theorem min_steps_ladder (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, (∃ (x y : ℕ), x * a - y * b = m ∧ x * a - y * b = 0) → m ≥ n :=
sorry

end min_steps_ladder_l3974_397430


namespace binary_repr_25_l3974_397480

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

theorem binary_repr_25 : binary_repr 25 = [true, false, false, true, true] := by sorry

end binary_repr_25_l3974_397480


namespace square_difference_l3974_397481

theorem square_difference (x y : ℚ) 
  (sum_eq : x + y = 9/13) 
  (diff_eq : x - y = 5/13) : 
  x^2 - y^2 = 45/169 := by
sorry

end square_difference_l3974_397481


namespace equilateral_triangle_circumcircle_parallel_lines_collinear_l3974_397421

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if a line is parallel to another line -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Finds the intersection point of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem equilateral_triangle_circumcircle_parallel_lines_collinear 
  (abc : Triangle) 
  (circ : Circle) 
  (p : Point) 
  (h1 : isEquilateral abc) 
  (h2 : pointOnCircle p circ) 
  (l1 l2 l3 : Line) 
  (h3 : isParallel l1 (Line.mk 1 0 0))  -- Parallel to BC
  (h4 : isParallel l2 (Line.mk 0 1 0))  -- Parallel to CA
  (h5 : isParallel l3 (Line.mk 1 1 0))  -- Parallel to AB
  : 
  let m := lineIntersection l2 (Line.mk 0 1 0)  -- Intersection with CA
  let n := lineIntersection l3 (Line.mk 1 1 0)  -- Intersection with AB
  let q := lineIntersection l1 (Line.mk 1 0 0)  -- Intersection with BC
  areCollinear m n q := by sorry

end equilateral_triangle_circumcircle_parallel_lines_collinear_l3974_397421


namespace cos_alpha_plus_5pi_12_l3974_397482

theorem cos_alpha_plus_5pi_12 (α : Real) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end cos_alpha_plus_5pi_12_l3974_397482


namespace trajectory_of_point_P_l3974_397483

/-- Given point A(1, 0) and line l: y = 2x - 4, with point R on line l such that
    vector RA equals vector AP, prove that the trajectory of point P is y = 2x -/
theorem trajectory_of_point_P (R P : ℝ × ℝ) :
  (∃ a : ℝ, R = (a, 2 * a - 4)) →  -- R is on line l: y = 2x - 4
  (R.1 - 1, R.2) = (P.1 - 1, P.2) →  -- vector RA = vector AP
  P.2 = 2 * P.1 :=  -- trajectory of P is y = 2x
by sorry

end trajectory_of_point_P_l3974_397483


namespace soil_cost_per_cubic_foot_l3974_397438

/-- Calculates the cost per cubic foot of soil for Bob's gardening project. -/
theorem soil_cost_per_cubic_foot
  (rose_bushes : ℕ)
  (rose_bush_cost : ℚ)
  (gardener_hourly_rate : ℚ)
  (gardener_hours_per_day : ℕ)
  (gardener_days : ℕ)
  (soil_volume : ℕ)
  (total_project_cost : ℚ)
  (h1 : rose_bushes = 20)
  (h2 : rose_bush_cost = 150)
  (h3 : gardener_hourly_rate = 30)
  (h4 : gardener_hours_per_day = 5)
  (h5 : gardener_days = 4)
  (h6 : soil_volume = 100)
  (h7 : total_project_cost = 4100) :
  (total_project_cost - (rose_bushes * rose_bush_cost + gardener_hourly_rate * gardener_hours_per_day * gardener_days)) / soil_volume = 5 := by
  sorry

end soil_cost_per_cubic_foot_l3974_397438


namespace ghee_mixture_quantity_l3974_397491

theorem ghee_mixture_quantity (original_ghee_percent : Real) 
                               (original_vanaspati_percent : Real)
                               (original_palm_oil_percent : Real)
                               (added_ghee : Real)
                               (added_palm_oil : Real)
                               (final_vanaspati_percent : Real) :
  original_ghee_percent = 0.55 →
  original_vanaspati_percent = 0.35 →
  original_palm_oil_percent = 0.10 →
  added_ghee = 15 →
  added_palm_oil = 5 →
  final_vanaspati_percent = 0.30 →
  ∃ (original_quantity : Real),
    original_quantity = 120 ∧
    original_vanaspati_percent * original_quantity = 
      final_vanaspati_percent * (original_quantity + added_ghee + added_palm_oil) :=
by sorry

end ghee_mixture_quantity_l3974_397491


namespace no_divisor_3_mod_4_and_unique_solution_l3974_397424

theorem no_divisor_3_mod_4_and_unique_solution : 
  (∀ x : ℤ, ∀ d : ℤ, d ∣ (x^2 + 1) → d % 4 ≠ 3) ∧ 
  (∀ x y : ℕ, x^2 - y^3 = 7 ↔ x = 23 ∧ y = 8) := by
  sorry

end no_divisor_3_mod_4_and_unique_solution_l3974_397424


namespace total_age_problem_l3974_397490

theorem total_age_problem (a b c : ℕ) : 
  b = 10 →
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 := by
sorry

end total_age_problem_l3974_397490


namespace arnold_protein_consumption_l3974_397400

def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def arnold_consumption (collagen_scoops protein_scoops : ℕ) : ℕ :=
  (collagen_scoops * collagen_protein_per_2_scoops / 2) + 
  (protein_scoops * protein_powder_per_scoop) + 
  steak_protein

theorem arnold_protein_consumption : 
  arnold_consumption 1 1 = 86 := by
  sorry

end arnold_protein_consumption_l3974_397400


namespace arithmetic_geometric_sequence_properties_l3974_397436

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1_bounds : 1 < a 1 ∧ a 1 < 3)
  (h_a3 : a 3 = 4)
  (b : ℕ → ℝ)
  (h_b_def : ∀ n, b n = 2^(a n)) :
  (∃ r, ∀ n, b (n + 1) = r * b n) ∧
  (b 1 < b 2) ∧
  (b 2 > 4) ∧
  (b 2 * b 4 = 256) :=
sorry

end arithmetic_geometric_sequence_properties_l3974_397436


namespace division_evaluation_l3974_397440

theorem division_evaluation : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by sorry

end division_evaluation_l3974_397440


namespace race_distance_proof_l3974_397463

/-- The total distance Jesse and Mia each need to run in a week-long race. -/
def total_distance : ℝ := 48

theorem race_distance_proof (jesse_first_three : ℝ) (jesse_day_four : ℝ) (mia_first_four : ℝ) (final_three_avg : ℝ) :
  jesse_first_three = 3 * (2/3) →
  jesse_day_four = 10 →
  mia_first_four = 4 * 3 →
  final_three_avg = 6 →
  total_distance = jesse_first_three + jesse_day_four + (3 * 2 * final_three_avg) / 2 :=
by sorry

end race_distance_proof_l3974_397463


namespace absolute_value_inequality_l3974_397406

theorem absolute_value_inequality (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (-8 ≤ x ∧ x < -5) ∨ (1 < x ∧ x ≤ 4) := by sorry

end absolute_value_inequality_l3974_397406


namespace circle_radii_sum_l3974_397409

theorem circle_radii_sum : 
  ∀ s : ℝ, 
  (s > 0) →
  (s^2 - 12*s + 12 = 0) →
  (∃ t : ℝ, s^2 - 12*s + 12 = 0 ∧ s ≠ t) →
  (s + (12 - s) = 12) := by
  sorry

end circle_radii_sum_l3974_397409


namespace last_digit_of_2008_power_last_digit_of_2008_to_2008_l3974_397426

theorem last_digit_of_2008_power (n : ℕ) : n > 0 → (2008^n) % 10 = (2008^(n % 4)) % 10 := by sorry

theorem last_digit_of_2008_to_2008 : (2008^2008) % 10 = 6 := by sorry

end last_digit_of_2008_power_last_digit_of_2008_to_2008_l3974_397426


namespace square_plus_self_divisible_by_two_l3974_397455

theorem square_plus_self_divisible_by_two (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end square_plus_self_divisible_by_two_l3974_397455


namespace intersection_with_complement_l3974_397452

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l3974_397452


namespace negation_of_existence_power_of_two_less_than_1000_l3974_397469

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem power_of_two_less_than_1000 :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) :=
by sorry

end negation_of_existence_power_of_two_less_than_1000_l3974_397469


namespace evaluate_expression_l3974_397429

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l3974_397429


namespace min_connections_for_six_towns_l3974_397476

/-- The number of towns -/
def num_towns : ℕ := 6

/-- The formula for the number of connections in an undirected graph without loops -/
def connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 towns, the minimum number of connections needed is 15 -/
theorem min_connections_for_six_towns :
  connections num_towns = 15 := by sorry

end min_connections_for_six_towns_l3974_397476


namespace absolute_value_inequality_solution_set_l3974_397484

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| ≤ 5} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} := by sorry

end absolute_value_inequality_solution_set_l3974_397484


namespace quadratic_function_properties_l3974_397431

-- Define the quadratic function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the maximum value function h(a)
noncomputable def h (a b c : ℝ) : ℝ := 
  let x₀ := -b / (2 * a)
  f a b c x₀

-- Main theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x)
  (hz : ∃! x, f a b c x + 6 * a = 0)
  : 
  (a = -1/5 ∧ b = -6/5 ∧ c = -3/5) ∧
  (∀ a' b' c', h a' b' c' ≥ -2) ∧
  (∃ a₀ b₀ c₀, h a₀ b₀ c₀ = -2)
  := by sorry

end quadratic_function_properties_l3974_397431


namespace scientific_notation_141260_l3974_397401

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_141260 :
  toScientificNotation 141260 = ScientificNotation.mk 1.4126 5 (by norm_num) :=
sorry

end scientific_notation_141260_l3974_397401


namespace frank_maze_time_l3974_397499

/-- Represents the maximum additional time Frank can spend in the current maze -/
def max_additional_time (current_time : ℕ) (previous_mazes : ℕ) (average_previous : ℕ) (max_average : ℕ) : ℕ :=
  max_average * (previous_mazes + 1) - (average_previous * previous_mazes + current_time)

/-- Theorem stating the maximum additional time Frank can spend in the maze -/
theorem frank_maze_time : max_additional_time 45 4 50 60 = 55 := by
  sorry

end frank_maze_time_l3974_397499


namespace average_difference_l3974_397478

def even_integers_16_to_44 : List Int :=
  List.range 15 |> List.map (fun i => 16 + 2 * i)

def even_integers_14_to_56 : List Int :=
  List.range 22 |> List.map (fun i => 14 + 2 * i)

def average (l : List Int) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_difference :
  average even_integers_16_to_44 + 5 = average even_integers_14_to_56 := by
  sorry

end average_difference_l3974_397478


namespace circle_properties_l3974_397460

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  3*x + 4*y - 6 = 0

def line2_equation (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Define the theorem
theorem circle_properties :
  -- Part 1: The circle exists when m < 5
  (∀ m : ℝ, m < 5 → ∃ x y : ℝ, circle_equation x y m) ∧
  -- Part 2: When the circle intersects line1 at M and N with |MN| = 2√3, m = 1
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line1_equation x₁ y₁ ∧
    line1_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 ∧
    m = 1) ∧
  -- Part 3: When the circle intersects line2 at A and B, there exists m = -2
  -- such that the circle with diameter AB passes through the origin
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line2_equation x₁ y₁ ∧
    line2_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = -2) := by
  sorry


end circle_properties_l3974_397460


namespace part1_solution_set_part2_m_range_l3974_397467

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part2_m_range :
  {m : ℝ | ∃ x, f m x ≤ 2*m - 5} = {m : ℝ | m ≥ 8} := by sorry

end part1_solution_set_part2_m_range_l3974_397467


namespace response_rate_percentage_l3974_397462

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : 
  responses_needed = 900 → questionnaires_mailed = 1500 → 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
sorry

end response_rate_percentage_l3974_397462


namespace chess_tournament_players_l3974_397457

/-- The number of players in the chess tournament -/
def num_players : ℕ := 45

/-- The total score of all players in the tournament -/
def total_score : ℕ := 1980

/-- Theorem stating that the number of players is correct given the total score -/
theorem chess_tournament_players :
  num_players * (num_players - 1) = total_score :=
by sorry

end chess_tournament_players_l3974_397457


namespace parabola_intersection_difference_l3974_397442

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 2 * x^2 - 4 * x + 4 = -x^2 - 2 * x + 4 → x = a ∨ x = c) ∧
  c ≥ a ∧
  c - a = 2/3 := by
  sorry

end parabola_intersection_difference_l3974_397442


namespace sum_of_distinct_remainders_for_ten_l3974_397451

theorem sum_of_distinct_remainders_for_ten : ∃ (s : Finset ℕ), 
  (∀ r ∈ s, ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ r = 10 % d) ∧ 
  (∀ d : ℕ, 1 ≤ d → d ≤ 9 → (10 % d) ∈ s) ∧
  (s.sum id = 10) := by
  sorry

end sum_of_distinct_remainders_for_ten_l3974_397451


namespace bees_flew_in_l3974_397454

theorem bees_flew_in (initial_bees final_bees : ℕ) (h : initial_bees ≤ final_bees) :
  final_bees - initial_bees = final_bees - initial_bees :=
by sorry

end bees_flew_in_l3974_397454


namespace multiplication_and_subtraction_l3974_397411

theorem multiplication_and_subtraction : 10 * (5 - 2) = 30 := by
  sorry

end multiplication_and_subtraction_l3974_397411


namespace travelers_checks_worth_l3974_397412

-- Define the problem parameters
def total_checks : ℕ := 30
def small_denomination : ℕ := 50
def large_denomination : ℕ := 100
def spent_checks : ℕ := 18
def remaining_average : ℕ := 75

-- Define the theorem
theorem travelers_checks_worth :
  ∀ (x y : ℕ),
    x + y = total_checks →
    x ≥ spent_checks →
    (small_denomination * (x - spent_checks) + large_denomination * y) / (total_checks - spent_checks) = remaining_average →
    small_denomination * x + large_denomination * y = 1800 :=
by
  sorry

end travelers_checks_worth_l3974_397412


namespace matrix_N_satisfies_conditions_l3974_397498

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![2, -1, 6; 3, 4, 0; -1, 1, -3]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 2; -5] + !![1; 1; 4] ∧
  N * j = !![-1; 4; 1] ∧
  N * k = !![6; 0; -3] :=
by sorry

end matrix_N_satisfies_conditions_l3974_397498


namespace binary_101_is_5_l3974_397418

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₂ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₂ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end binary_101_is_5_l3974_397418


namespace dereks_score_l3974_397410

/-- Given a basketball team's score and the performance of other players, 
    calculate Derek's score. -/
theorem dereks_score 
  (total_score : ℕ) 
  (other_players : ℕ) 
  (avg_score_others : ℕ) 
  (h1 : total_score = 65) 
  (h2 : other_players = 8) 
  (h3 : avg_score_others = 5) : 
  total_score - (other_players * avg_score_others) = 25 := by
sorry

end dereks_score_l3974_397410


namespace min_value_quadratic_l3974_397416

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end min_value_quadratic_l3974_397416


namespace cube_sum_product_l3974_397403

def is_even_or_prime (n : ℕ) : Prop :=
  Even n ∨ Nat.Prime n

theorem cube_sum_product : ∃ (a b : ℕ), 
  a^3 + b^3 = 91 ∧ 
  is_even_or_prime a ∧ 
  is_even_or_prime b ∧ 
  a * b = 12 := by sorry

end cube_sum_product_l3974_397403


namespace homework_problem_count_l3974_397450

theorem homework_problem_count (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  p * t = (2 * p - 10) * (t - 1) → 
  p * t = 60 := by
  sorry

end homework_problem_count_l3974_397450


namespace fourth_buoy_adjusted_distance_l3974_397422

def buoy_distance (n : ℕ) : ℝ :=
  20 + 4 * (n - 1)

def ocean_current : ℝ := 3

theorem fourth_buoy_adjusted_distance :
  buoy_distance 4 - ocean_current = 29 := by
  sorry

end fourth_buoy_adjusted_distance_l3974_397422


namespace least_addition_for_multiple_of_five_l3974_397439

theorem least_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (879 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (879 + m) % 5 = 0 → n ≤ m :=
by sorry

end least_addition_for_multiple_of_five_l3974_397439


namespace tan_triple_angle_l3974_397492

theorem tan_triple_angle (α : Real) (P : ℝ × ℝ) :
  α > 0 ∧ α < π / 2 →  -- α is acute
  P.1 = 2 * (Real.cos (280 * π / 180))^2 →  -- x-coordinate of P
  P.2 = Real.sin (20 * π / 180) →  -- y-coordinate of P
  Real.tan (3 * α) = Real.sqrt 3 := by
    sorry

end tan_triple_angle_l3974_397492


namespace power_of_three_squared_l3974_397413

theorem power_of_three_squared : 3^2 = 9 := by sorry

end power_of_three_squared_l3974_397413


namespace washing_machine_last_load_l3974_397456

theorem washing_machine_last_load (capacity : ℕ) (total_clothes : ℕ) : 
  capacity = 28 → total_clothes = 200 → 
  total_clothes % capacity = 4 :=
by
  sorry

end washing_machine_last_load_l3974_397456


namespace gcd_lcm_multiple_relation_l3974_397473

theorem gcd_lcm_multiple_relation (x y z : ℤ) (h1 : y ≠ 0) (h2 : x / y = z) : 
  Int.gcd x y = y ∧ Int.lcm x y = x := by sorry

end gcd_lcm_multiple_relation_l3974_397473


namespace divisors_in_range_l3974_397496

theorem divisors_in_range (m a b : ℕ) (hm : 0 < m) (ha : m^2 < a) (hb : m^2 < b) 
  (ha_upper : a < m^2 + m) (hb_upper : b < m^2 + m) (hab : a ≠ b) : 
  ∀ d : ℕ, m^2 < d → d < m^2 + m → d ∣ (a * b) → d = a ∨ d = b := by
sorry

end divisors_in_range_l3974_397496


namespace circle_chord_length_l3974_397459

theorem circle_chord_length (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0 → (∃ x₀ y₀ : ℝ, x₀ + y₀ + 4 = 0 ∧ 
    (x - x₀)^2 + (y - y₀)^2 = 2^2)) → 
  a = -7 := by
sorry


end circle_chord_length_l3974_397459


namespace mike_total_games_l3974_397471

/-- The number of video games Mike had initially -/
def total_games : ℕ := sorry

/-- The number of non-working games -/
def non_working_games : ℕ := 8

/-- The price of each working game in dollars -/
def price_per_game : ℕ := 7

/-- The total amount earned from selling working games in dollars -/
def total_earned : ℕ := 56

/-- Theorem stating that the total number of video games Mike had initially is 16 -/
theorem mike_total_games : total_games = 16 := by sorry

end mike_total_games_l3974_397471


namespace quadratic_expression_value_l3974_397448

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3*x + y = 5) 
  (h2 : x + 3*y = 6) : 
  10*x^2 + 13*x*y + 10*y^2 = 97 := by
  sorry

end quadratic_expression_value_l3974_397448


namespace intersection_equals_subset_implies_union_equals_set_l3974_397487

theorem intersection_equals_subset_implies_union_equals_set 
  (M P : Set α) (h : M ∩ P = P) : M ∪ P = M := by
  sorry

end intersection_equals_subset_implies_union_equals_set_l3974_397487


namespace negation_of_proposition_negation_of_inequality_l3974_397465

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) := by sorry

theorem negation_of_inequality :
  (¬∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end negation_of_proposition_negation_of_inequality_l3974_397465


namespace line_touches_ellipse_l3974_397447

theorem line_touches_ellipse (a b : ℝ) (m : ℝ) (h1 : a = 3) (h2 : b = 1) :
  (∃! p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 1/3 :=
sorry

end line_touches_ellipse_l3974_397447
