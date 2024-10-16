import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2234_223447

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 30)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2234_223447


namespace NUMINAMATH_CALUDE_original_number_of_professors_l2234_223439

theorem original_number_of_professors : 
  ∃ p : ℕ+, 
    p.val > 0 ∧ 
    6480 % p.val = 0 ∧ 
    11200 % (p.val + 3) = 0 ∧ 
    (6480 : ℚ) / p.val < (11200 : ℚ) / (p.val + 3) ∧ 
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_professors_l2234_223439


namespace NUMINAMATH_CALUDE_fair_die_probabilities_l2234_223413

-- Define the sample space for a fair six-sided die
def Ω : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define event A: number of points ≥ 3
def A : Finset ℕ := {3, 4, 5, 6}

-- Define event B: number of points is odd
def B : Finset ℕ := {1, 3, 5}

-- Define the probability measure for a fair die
def P (S : Finset ℕ) : ℚ := (S ∩ Ω).card / Ω.card

-- Theorem statement
theorem fair_die_probabilities :
  P A = 2/3 ∧ P (A ∪ B) = 5/6 ∧ P (A ∩ B) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_probabilities_l2234_223413


namespace NUMINAMATH_CALUDE_distinct_integer_roots_l2234_223423

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  (a = 4.5 ∨ a = 1 ∨ a = -9 ∨ a = -12.5) :=
sorry

end NUMINAMATH_CALUDE_distinct_integer_roots_l2234_223423


namespace NUMINAMATH_CALUDE_concert_attendance_l2234_223457

theorem concert_attendance (total_tickets : ℕ) 
  (h1 : total_tickets = 2465)
  (before_start : ℕ) 
  (h2 : before_start = (7 * total_tickets) / 8)
  (after_first_song : ℕ) 
  (h3 : after_first_song = (13 * (total_tickets - before_start)) / 17)
  (last_performances : ℕ) 
  (h4 : last_performances = 47) : 
  total_tickets - before_start - after_first_song - last_performances = 26 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l2234_223457


namespace NUMINAMATH_CALUDE_equation_solution_range_l2234_223467

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*x - m) / (x - 3) - 1 = x / (3 - x)) → 
  (m > 3 ∧ m ≠ 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2234_223467


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l2234_223436

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_plane 
  (α β : Plane) (m n : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_perp_β : perpendicular_line_plane m β)
  (h_n_para_β : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l2234_223436


namespace NUMINAMATH_CALUDE_sailboat_problem_l2234_223494

theorem sailboat_problem (small_sail_size : ℝ) (small_sail_speed : ℝ) 
  (big_sail_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  small_sail_size = 12 →
  small_sail_speed = 20 →
  big_sail_speed = 50 →
  distance = 200 →
  time_difference = 6 →
  distance / small_sail_speed - distance / big_sail_speed = time_difference →
  ∃ big_sail_size : ℝ, 
    big_sail_size = 30 ∧ 
    small_sail_speed / big_sail_speed = small_sail_size / big_sail_size :=
by sorry


end NUMINAMATH_CALUDE_sailboat_problem_l2234_223494


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l2234_223464

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l2234_223464


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l2234_223435

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 20) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 4) : 
  stripe_width * revolutions * (π * diameter) = 640 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l2234_223435


namespace NUMINAMATH_CALUDE_nick_sold_fewer_bottles_l2234_223465

/-- Proves that Nick sold 6 fewer bottles of soda than Remy in the morning -/
theorem nick_sold_fewer_bottles (remy_morning : ℕ) (price : ℚ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  price = 1/2 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (nick_morning : ℕ), remy_morning - nick_morning = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_nick_sold_fewer_bottles_l2234_223465


namespace NUMINAMATH_CALUDE_steve_answerable_questions_l2234_223481

theorem steve_answerable_questions (total_questions : ℕ) (difference : ℕ) : 
  total_questions = 45 → difference = 7 → total_questions - difference = 38 := by
sorry

end NUMINAMATH_CALUDE_steve_answerable_questions_l2234_223481


namespace NUMINAMATH_CALUDE_butterfingers_count_l2234_223458

theorem butterfingers_count (total : ℕ) (mars : ℕ) (snickers : ℕ) (butterfingers : ℕ)
  (h1 : total = 12)
  (h2 : mars = 2)
  (h3 : snickers = 3)
  (h4 : total = mars + snickers + butterfingers) :
  butterfingers = 7 := by
  sorry

end NUMINAMATH_CALUDE_butterfingers_count_l2234_223458


namespace NUMINAMATH_CALUDE_special_sequence_value_of_2_special_sequence_verification_l2234_223462

/-- A sequence where each term n is mapped to 6n, except for 6 which maps to 1 -/
def special_sequence : ℕ → ℕ
| 6 => 1
| n => 6 * n

/-- The theorem states that the value corresponding to 2 in the special sequence is 12 -/
theorem special_sequence_value_of_2 : special_sequence 2 = 12 := by
  sorry

/-- Verification of other given values in the sequence -/
theorem special_sequence_verification :
  special_sequence 1 = 6 ∧
  special_sequence 3 = 18 ∧
  special_sequence 4 = 24 ∧
  special_sequence 5 = 30 ∧
  special_sequence 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_value_of_2_special_sequence_verification_l2234_223462


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2234_223408

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 800 ∧ cost_price = 640 →
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2234_223408


namespace NUMINAMATH_CALUDE_yard_length_22_trees_l2234_223414

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 22 trees planted at equal distances,
    with one tree at each end and 21 metres between consecutive trees, is 441 metres. -/
theorem yard_length_22_trees : yard_length 22 21 = 441 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_22_trees_l2234_223414


namespace NUMINAMATH_CALUDE_handshake_arrangement_theorem_l2234_223426

-- Define the number of people in the group
def num_people : ℕ := 12

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := 3

-- Define the function to calculate the number of distinct handshaking arrangements
def num_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Define the function to calculate the remainder when divided by 1000
def remainder_mod_1000 (x : ℕ) : ℕ := x % 1000

-- Theorem statement
theorem handshake_arrangement_theorem :
  num_arrangements num_people handshakes_per_person = 680680 ∧
  remainder_mod_1000 (num_arrangements num_people handshakes_per_person) = 680 := by
  sorry

end NUMINAMATH_CALUDE_handshake_arrangement_theorem_l2234_223426


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l2234_223482

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The composition of reflecting over y-axis and then x-axis -/
def reflect_yx (p : ℝ × ℝ) : ℝ × ℝ := reflect_x (reflect_y p)

theorem reflect_F_coordinates :
  reflect_yx (6, -4) = (-6, 4) := by sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l2234_223482


namespace NUMINAMATH_CALUDE_egg_collection_total_l2234_223410

/-- The number of dozen eggs Benjamin collects -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_total : total_eggs = 26 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_total_l2234_223410


namespace NUMINAMATH_CALUDE_subtraction_problem_l2234_223411

theorem subtraction_problem : 444 - 44 - 4 = 396 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2234_223411


namespace NUMINAMATH_CALUDE_smallest_value_x_squared_plus_8x_l2234_223427

theorem smallest_value_x_squared_plus_8x :
  (∀ x : ℝ, x^2 + 8*x ≥ -16) ∧ (∃ x : ℝ, x^2 + 8*x = -16) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_x_squared_plus_8x_l2234_223427


namespace NUMINAMATH_CALUDE_sum_of_two_primes_odd_implies_one_is_two_l2234_223422

theorem sum_of_two_primes_odd_implies_one_is_two (p q : ℕ) :
  Prime p → Prime q → Odd (p + q) → (p = 2 ∨ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_odd_implies_one_is_two_l2234_223422


namespace NUMINAMATH_CALUDE_pentagon_smallest_angle_l2234_223469

theorem pentagon_smallest_angle (a b c d e : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b + c + d + e = 540 →
  b = 4/3 * a →
  c = 5/3 * a →
  d = 2 * a →
  e = 7/3 * a →
  a = 64.8 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_smallest_angle_l2234_223469


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2234_223453

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector of B
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angleMeasure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_bisector_length 
  (t : Triangle) 
  (h1 : angleMeasure t.A t.B t.C = 20)
  (h2 : angleMeasure t.C t.A t.B = 40)
  (h3 : length t.A t.C - length t.A t.B = 5) :
  length t.B (angleBisector t) = 5 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l2234_223453


namespace NUMINAMATH_CALUDE_absolute_value_at_two_l2234_223470

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  ∃ (a b c d : ℝ), ∀ x, g x = a * x^3 + b * x^2 + c * x + d ∧
  (|g 0| = 10) ∧ (|g 1| = 10) ∧ (|g 3| = 10) ∧
  (|g 4| = 10) ∧ (|g 5| = 10) ∧ (|g 8| = 10)

/-- Theorem stating that if g satisfies the condition, then |g(2)| = 20 -/
theorem absolute_value_at_two
  (g : ThirdDegreePolynomial)
  (h : SatisfiesCondition g) :
  |g 2| = 20 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_at_two_l2234_223470


namespace NUMINAMATH_CALUDE_complex_square_imaginary_part_l2234_223417

theorem complex_square_imaginary_part : 
  ∃ (a b : ℝ), (1 + Complex.I)^2 = (a : ℂ) + (b : ℂ) * Complex.I → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_imaginary_part_l2234_223417


namespace NUMINAMATH_CALUDE_joint_purchase_popularity_l2234_223476

structure JointPurchase where
  scale : ℝ
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience : ℝ
  dispute_potential : ℝ

def benefits (jp : JointPurchase) : ℝ :=
  jp.cost_savings + jp.quality_assessment + jp.community_trust

def drawbacks (jp : JointPurchase) : ℝ :=
  jp.transaction_costs + jp.organizational_efforts + jp.convenience + jp.dispute_potential

theorem joint_purchase_popularity (jp : JointPurchase) :
  jp.scale > 1 → benefits jp > drawbacks jp ∧
  jp.scale ≤ 1 → benefits jp ≤ drawbacks jp :=
sorry

end NUMINAMATH_CALUDE_joint_purchase_popularity_l2234_223476


namespace NUMINAMATH_CALUDE_number_divided_by_three_l2234_223480

theorem number_divided_by_three (x : ℝ) (h : x - 39 = 54) : x / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l2234_223480


namespace NUMINAMATH_CALUDE_angle_WYZ_measure_l2234_223428

-- Define the angles
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 25 := by sorry

end NUMINAMATH_CALUDE_angle_WYZ_measure_l2234_223428


namespace NUMINAMATH_CALUDE_triangle_formation_l2234_223463

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three real numbers can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 1 1.5 3 ∧
  ¬ can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l2234_223463


namespace NUMINAMATH_CALUDE_obtuse_triangle_equilateral_triangle_l2234_223407

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem 1: If cos A * cos B * cos C < 0, then the triangle is obtuse
theorem obtuse_triangle (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 →
  (t.A > π/2 ∨ t.B > π/2 ∨ t.C > π/2) :=
sorry

-- Theorem 2: If cos(A-C) * cos(B-C) * cos(C-A) = 1, then the triangle is equilateral
theorem equilateral_triangle (t : Triangle) :
  Real.cos (t.A - t.C) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 →
  t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_equilateral_triangle_l2234_223407


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l2234_223498

/-- Given positive real numbers x and y that vary inversely with respect to x^3,
    prove that if y = 8 when x = 2, then x = 0.4 when y = 1000. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k) 
    (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 1000 → x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l2234_223498


namespace NUMINAMATH_CALUDE_decimal_addition_l2234_223472

theorem decimal_addition : (0.9 : ℝ) + 0.99 = 1.89 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l2234_223472


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l2234_223489

-- Define a plane
structure Plane :=
  (α : Type*)

-- Define a line
structure Line :=
  (l : Type*)

-- Define perpendicular relation between a line and a plane
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

-- Define a line being contained within a plane
def contained_in_plane (m : Line) (α : Plane) : Prop :=
  sorry

-- Define perpendicular relation between two lines
def perpendicular_lines (l m : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane
  (l m : Line) (α : Plane)
  (h1 : perpendicular_to_plane l α)
  (h2 : contained_in_plane m α) :
  perpendicular_lines l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l2234_223489


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2234_223450

def f (x : ℝ) : ℝ := |x - 1|

theorem f_increasing_on_interval : 
  ∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2234_223450


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2234_223452

theorem integer_root_of_cubic (a b c : ℚ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (f (3 - Real.sqrt 5) = 0) →
  (∃ r : ℤ, f r = 0) →
  (∃ r : ℤ, f r = 0 ∧ r = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2234_223452


namespace NUMINAMATH_CALUDE_dilution_correct_l2234_223442

/-- The amount of pure alcohol needed to dilute iodine tincture -/
def alcohol_amount : ℝ := 2275

/-- The initial amount of iodine tincture in grams -/
def initial_tincture : ℝ := 350

/-- The initial iodine content as a percentage -/
def initial_content : ℝ := 15

/-- The desired iodine content as a percentage -/
def desired_content : ℝ := 2

/-- Theorem stating that adding the calculated amount of alcohol results in the desired iodine content -/
theorem dilution_correct : 
  (initial_tincture * initial_content) / (initial_tincture + alcohol_amount) = desired_content := by
  sorry

end NUMINAMATH_CALUDE_dilution_correct_l2234_223442


namespace NUMINAMATH_CALUDE_candidate_a_democratic_votes_l2234_223440

theorem candidate_a_democratic_votes 
  (total_voters : ℝ) 
  (dem_percent : ℝ) 
  (rep_percent : ℝ) 
  (rep_for_a_percent : ℝ) 
  (total_for_a_percent : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 0.4 →
  rep_for_a_percent = 0.2 →
  total_for_a_percent = 0.59 →
  ∃ (dem_for_a_percent : ℝ),
    dem_for_a_percent * dem_percent * total_voters + 
    rep_for_a_percent * rep_percent * total_voters = 
    total_for_a_percent * total_voters ∧
    dem_for_a_percent = 0.85 :=
by sorry

end NUMINAMATH_CALUDE_candidate_a_democratic_votes_l2234_223440


namespace NUMINAMATH_CALUDE_third_month_sale_l2234_223418

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sales : List ℕ := [6400, 7000, 7200, 6500, 5100]

theorem third_month_sale :
  (num_months * average_sale - sales.sum) = 6800 :=
sorry

end NUMINAMATH_CALUDE_third_month_sale_l2234_223418


namespace NUMINAMATH_CALUDE_problem_statement_l2234_223415

open Real

variable (a b : ℝ)

theorem problem_statement (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) > 4 ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → sqrt (1 + a) + sqrt (1 + b) ≤ sqrt 6) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (a * b + 4 * a + b) / (4 * a + b) ≤ 10 / 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2234_223415


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l2234_223401

theorem smallest_positive_solution (x : ℝ) :
  (x > 0 ∧ x / 7 + 2 / (7 * x) = 1) → x = (7 - Real.sqrt 41) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l2234_223401


namespace NUMINAMATH_CALUDE_ratio_of_two_numbers_l2234_223434

theorem ratio_of_two_numbers (x y : ℝ) (h1 : x + y = 33) (h2 : x = 22) :
  y / x = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_of_two_numbers_l2234_223434


namespace NUMINAMATH_CALUDE_largest_quantity_l2234_223461

theorem largest_quantity (x y z w : ℝ) 
  (h : x + 5 = y - 3 ∧ x + 5 = z + 2 ∧ x + 5 = w - 4) : 
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2234_223461


namespace NUMINAMATH_CALUDE_expression_meaning_l2234_223404

theorem expression_meaning (a : ℝ) : 2 * (a - 3)^2 = 2 * (a - 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_meaning_l2234_223404


namespace NUMINAMATH_CALUDE_geometric_progression_sufficient_not_necessary_l2234_223477

/-- A sequence of three real numbers forms a geometric progression --/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_progression_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_progression a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_progression a b c) := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_sufficient_not_necessary_l2234_223477


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_six_l2234_223499

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_power_of_six :
  tens_digit (last_two_digits ((4 + 2)^21)) + ones_digit (last_two_digits ((4 + 2)^21)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_six_l2234_223499


namespace NUMINAMATH_CALUDE_purely_imaginary_number_l2234_223412

theorem purely_imaginary_number (k : ℝ) : 
  (∃ (z : ℂ), z = (2 * k^2 - 3 * k - 2 : ℝ) + (k^2 - 2 * k : ℝ) * I ∧ z.re = 0 ∧ z.im ≠ 0) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_number_l2234_223412


namespace NUMINAMATH_CALUDE_altitude_least_integer_l2234_223455

theorem altitude_least_integer (a b : ℝ) (h : a = 5 ∧ b = 12) : 
  ∃ (L : ℝ), L = (a * b) / (2 * Real.sqrt (a^2 + b^2)) ∧ 
  (∀ (n : ℤ), (n : ℝ) > L → n ≥ 5) ∧ (4 : ℝ) < L :=
sorry

end NUMINAMATH_CALUDE_altitude_least_integer_l2234_223455


namespace NUMINAMATH_CALUDE_parallel_line_point_l2234_223483

/-- Given two points on a line and another line it's parallel to, prove the x-coordinate of the second point. -/
theorem parallel_line_point (j : ℝ) : 
  (∃ (m b : ℝ), (2 : ℝ) + 3 * m = -6 ∧ 
                 (19 : ℝ) - (-3) = m * (j - 4)) → 
  j = -29 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_point_l2234_223483


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2234_223433

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2234_223433


namespace NUMINAMATH_CALUDE_subset_condition_implies_p_range_l2234_223493

open Set

theorem subset_condition_implies_p_range (p : ℝ) : 
  let A : Set ℝ := {x | 4 * x + p < 0}
  let B : Set ℝ := {x | x < -1 ∨ x > 2}
  A.Nonempty → B.Nonempty → A ⊆ B → p ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_implies_p_range_l2234_223493


namespace NUMINAMATH_CALUDE_first_square_covering_all_rows_l2234_223468

-- Define a function to calculate the row number of a square
def row_number (n : ℕ) : ℕ := (n - 1) / 10 + 1

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Theorem statement
theorem first_square_covering_all_rows :
  (∀ r : ℕ, r ≥ 1 → r ≤ 10 → ∃ n : ℕ, n ≤ 100 ∧ is_perfect_square n ∧ row_number n = r) ∧
  (∀ m : ℕ, m < 100 → ¬(∀ r : ℕ, r ≥ 1 → r ≤ 10 → ∃ n : ℕ, n ≤ m ∧ is_perfect_square n ∧ row_number n = r)) :=
by sorry

end NUMINAMATH_CALUDE_first_square_covering_all_rows_l2234_223468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2234_223403

/-- Given an arithmetic sequence, prove that if the sum of the first n terms is 54
    and the sum of the first 2n terms is 72, then the sum of the first 3n terms is 78. -/
theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 54 → S (2*n) = 72 → S (3*n) = 78 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2234_223403


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l2234_223456

def modified_fibonacci_factorial_series : List Nat :=
  [1, 2, 3, 4, 7, 11, 18, 29, 47, 76]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_series :
  (modified_fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 73 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l2234_223456


namespace NUMINAMATH_CALUDE_ticket_sales_l2234_223459

theorem ticket_sales (adult_price children_price total_amount adult_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : children_price = 2)
  (h3 : total_amount = 275)
  (h4 : adult_tickets = 35) :
  ∃ children_tickets : ℕ, adult_tickets + children_tickets = 85 ∧ 
    adult_price * adult_tickets + children_price * children_tickets = total_amount :=
by sorry

end NUMINAMATH_CALUDE_ticket_sales_l2234_223459


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l2234_223441

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l2234_223441


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2234_223471

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N^3 - 3 • N^2 + 2 • N = !![2, 6; 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2234_223471


namespace NUMINAMATH_CALUDE_sigma_multiple_inequality_l2234_223448

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

theorem sigma_multiple_inequality (n : ℕ+) (h : sigma n > 2 * n) :
  ∀ m : ℕ+, (∃ k : ℕ+, m = k * n) → sigma m > 2 * m := by sorry

end NUMINAMATH_CALUDE_sigma_multiple_inequality_l2234_223448


namespace NUMINAMATH_CALUDE_range_equivalence_l2234_223432

/-- The range of real numbers a for which at least one of the given equations has real roots -/
def range_with_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
            (x^2 + (a-1)*x + a^2 = 0) ∨ 
            (x^2 + 2*a*x - 2*a = 0)

/-- The range of real numbers a for which none of the given equations have real roots -/
def range_without_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + 4*a*x - 4*a + 3 ≠ 0) ∧ 
            (x^2 + (a-1)*x + a^2 ≠ 0) ∧ 
            (x^2 + 2*a*x - 2*a ≠ 0)

/-- The theorem stating that the range with real roots is the complement of the range without real roots -/
theorem range_equivalence : 
  ∀ a : ℝ, range_with_real_roots a ↔ ¬(range_without_real_roots a) :=
sorry

end NUMINAMATH_CALUDE_range_equivalence_l2234_223432


namespace NUMINAMATH_CALUDE_solution_range_l2234_223429

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 2) + 1 = x / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ equation m x) ↔ (m ≤ 2 ∧ m ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2234_223429


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l2234_223402

theorem cos_two_theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4) : 
  Real.cos (2 * θ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l2234_223402


namespace NUMINAMATH_CALUDE_linear_relationship_l2234_223431

/-- Given a linear relationship where an increase of 4 units in x corresponds to an increase of 6 units in y,
    prove that an increase of 12 units in x will result in an increase of 18 units in y. -/
theorem linear_relationship (f : ℝ → ℝ) (x₀ : ℝ) :
  (f (x₀ + 4) - f x₀ = 6) → (f (x₀ + 12) - f x₀ = 18) := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_l2234_223431


namespace NUMINAMATH_CALUDE_arithmetic_mean_three_digit_multiples_of_seven_l2234_223479

/-- The smallest positive three-digit multiple of 7 -/
def smallest_multiple : ℕ := 105

/-- The largest positive three-digit multiple of 7 -/
def largest_multiple : ℕ := 994

/-- The number of positive three-digit multiples of 7 -/
def num_multiples : ℕ := 128

/-- The sum of all positive three-digit multiples of 7 -/
def sum_multiples : ℕ := 70336

theorem arithmetic_mean_three_digit_multiples_of_seven :
  (sum_multiples : ℚ) / (num_multiples : ℚ) = 549.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_three_digit_multiples_of_seven_l2234_223479


namespace NUMINAMATH_CALUDE_monomial_is_algebraic_expression_l2234_223496

-- Define what an algebraic expression is
def AlgebraicExpression (α : Type*) := α → ℝ

-- Define what a monomial is
def Monomial (α : Type*) := AlgebraicExpression α

-- Theorem: Every monomial is an algebraic expression
theorem monomial_is_algebraic_expression {α : Type*} :
  ∀ (m : Monomial α), ∃ (a : AlgebraicExpression α), m = a :=
sorry

end NUMINAMATH_CALUDE_monomial_is_algebraic_expression_l2234_223496


namespace NUMINAMATH_CALUDE_mnp_value_l2234_223409

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1))
  (h2 : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) :
  m * n * p = 12 := by
  sorry

end NUMINAMATH_CALUDE_mnp_value_l2234_223409


namespace NUMINAMATH_CALUDE_janes_number_l2234_223492

theorem janes_number (x : ℝ) : 5 * (2 * x + 15) = 175 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_janes_number_l2234_223492


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2234_223400

/-- The function f(x) = x * e^(-x) is increasing on (-∞, 1) -/
theorem f_increasing_on_interval (x : ℝ) : x < 1 → Monotone (fun x => x * Real.exp (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2234_223400


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2234_223475

def number_of_books : ℕ := 6
def number_of_identical_pairs : ℕ := 2
def books_per_pair : ℕ := 2

theorem book_arrangement_count :
  (number_of_books.factorial) / ((books_per_pair.factorial) ^ number_of_identical_pairs) = 180 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2234_223475


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2234_223444

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(13 * m.val) % 8 = 567 % 8) ∧ 
  (13 * n.val) % 8 = 567 % 8 → 
  n = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2234_223444


namespace NUMINAMATH_CALUDE_original_number_proof_l2234_223474

theorem original_number_proof :
  ∃! x : ℕ, (x + 2) % 17 = 0 ∧ x < 17 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2234_223474


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2234_223420

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2234_223420


namespace NUMINAMATH_CALUDE_wilson_payment_is_17_10_l2234_223491

/-- Calculates the total payment for Wilson's fast-food order --/
def wilsonPayment (hamburgerPrice fryPrice colaPrice sundaePrice couponDiscount loyaltyDiscount : ℚ) : ℚ :=
  let subtotal := 2 * hamburgerPrice + 3 * colaPrice + fryPrice + sundaePrice
  let afterCoupon := subtotal - couponDiscount
  afterCoupon * (1 - loyaltyDiscount)

/-- Theorem stating that Wilson's payment is $17.10 --/
theorem wilson_payment_is_17_10 :
  wilsonPayment 5 3 2 4 4 (1/10) = 171/10 := by
  sorry

end NUMINAMATH_CALUDE_wilson_payment_is_17_10_l2234_223491


namespace NUMINAMATH_CALUDE_B_power_60_is_identity_l2234_223430

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 0],
    ![0, 0, -1],
    ![0, 1, 0]]

theorem B_power_60_is_identity :
  B ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_60_is_identity_l2234_223430


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l2234_223484

theorem lawrence_county_kids_count :
  let group_a_1week : ℕ := 175000
  let group_a_2week : ℕ := 107000
  let group_a_3week : ℕ := 35000
  let group_b_1week : ℕ := 100000
  let group_b_2week : ℕ := 70350
  let group_b_3week : ℕ := 19500
  let group_c_1week : ℕ := 45000
  let group_c_2week : ℕ := 87419
  let group_c_3week : ℕ := 14425
  let kids_staying_home : ℕ := 590796
  let kids_outside_county : ℕ := 22
  
  let total_group_a : ℕ := group_a_1week + group_a_2week + group_a_3week
  let total_group_b : ℕ := group_b_1week + group_b_2week + group_b_3week
  let total_group_c : ℕ := group_c_1week + group_c_2week + group_c_3week
  
  let total_kids_in_camp : ℕ := total_group_a + total_group_b + total_group_c
  
  total_kids_in_camp + kids_staying_home + kids_outside_county = 1244512 :=
by
  sorry

#check lawrence_county_kids_count

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l2234_223484


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l2234_223419

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base 10 equivalent of 65432 in base 7 is 16340 --/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l2234_223419


namespace NUMINAMATH_CALUDE_pattern_result_l2234_223445

-- Define the pattern function
def pattern (a b : ℕ) : ℕ := sorry

-- Define the given operations
axiom op1 : pattern 3 7 = 27
axiom op2 : pattern 4 5 = 32
axiom op3 : pattern 5 8 = 60
axiom op4 : pattern 6 7 = 72
axiom op5 : pattern 7 8 = 98

-- Theorem to prove
theorem pattern_result : pattern 2 3 = 26 := by sorry

end NUMINAMATH_CALUDE_pattern_result_l2234_223445


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2234_223416

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- The number of factors of 10 in n! -/
def factorsOfTen (n : ℕ) : ℕ := sorry

theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 52 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2234_223416


namespace NUMINAMATH_CALUDE_salad_dressing_weight_l2234_223495

/-- Calculates the total weight of a salad dressing mixture --/
theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density : ℝ) :
  bowl_capacity = 150 ∧
  oil_fraction = 2/3 ∧
  vinegar_fraction = 1/3 ∧
  oil_density = 5 ∧
  vinegar_density = 4 →
  bowl_capacity * oil_fraction * oil_density +
  bowl_capacity * vinegar_fraction * vinegar_density = 700 := by
  sorry

end NUMINAMATH_CALUDE_salad_dressing_weight_l2234_223495


namespace NUMINAMATH_CALUDE_symmetry_of_point_wrt_x_axis_l2234_223437

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The X-axis symmetry operation -/
def xAxisSymmetry (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem symmetry_of_point_wrt_x_axis :
  let P : Point3D := ⟨-1, 8, 4⟩
  xAxisSymmetry P = ⟨-1, -8, 4⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetry_of_point_wrt_x_axis_l2234_223437


namespace NUMINAMATH_CALUDE_real_roots_imply_b_equals_one_l2234_223406

theorem real_roots_imply_b_equals_one (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*I*x + b = 1) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_imply_b_equals_one_l2234_223406


namespace NUMINAMATH_CALUDE_sum_of_cubes_minus_product_l2234_223421

theorem sum_of_cubes_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_minus_product_l2234_223421


namespace NUMINAMATH_CALUDE_optimal_tank_design_l2234_223460

/-- Represents the dimensions and cost of a rectangular open-top water storage tank. -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ
  base_cost : ℝ
  wall_cost : ℝ

/-- Calculates the volume of the tank. -/
def volume (t : Tank) : ℝ := t.length * t.width * t.depth

/-- Calculates the total construction cost of the tank. -/
def construction_cost (t : Tank) : ℝ :=
  t.base_cost * t.length * t.width + t.wall_cost * 2 * (t.length + t.width) * t.depth

/-- Theorem stating the optimal dimensions and minimum cost for the tank. -/
theorem optimal_tank_design :
  ∃ (t : Tank),
    t.depth = 3 ∧
    volume t = 4800 ∧
    t.base_cost = 150 ∧
    t.wall_cost = 120 ∧
    t.length = t.width ∧
    t.length = 40 ∧
    construction_cost t = 297600 ∧
    ∀ (t' : Tank),
      t'.depth = 3 →
      volume t' = 4800 →
      t'.base_cost = 150 →
      t'.wall_cost = 120 →
      construction_cost t' ≥ construction_cost t :=
by sorry

end NUMINAMATH_CALUDE_optimal_tank_design_l2234_223460


namespace NUMINAMATH_CALUDE_marble_game_probability_l2234_223486

theorem marble_game_probability (B R : ℕ) : 
  B + R = 21 →
  (B : ℚ) / 21 * ((B - 1) : ℚ) / 20 = 1 / 2 →
  B^2 + R^2 = 261 := by
sorry

end NUMINAMATH_CALUDE_marble_game_probability_l2234_223486


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2234_223446

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2234_223446


namespace NUMINAMATH_CALUDE_waiter_new_customers_l2234_223485

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 19)
  (h2 : customers_left = 14)
  (h3 : final_customers = 41) :
  final_customers - (initial_customers - customers_left) = 36 := by
  sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l2234_223485


namespace NUMINAMATH_CALUDE_min_value_condition_l2234_223449

open Set

variables {f : ℝ → ℝ} {a b : ℝ}

theorem min_value_condition (h_diff : Differentiable ℝ f) (h_cont : ContinuousOn f (Ioo a b)) :
  (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0) →
  (∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) ∧
  ¬ ((∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) →
     (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_condition_l2234_223449


namespace NUMINAMATH_CALUDE_radioactive_balls_identification_l2234_223454

/-- Represents a strategy for identifying radioactive balls -/
structure Strategy where
  num_tests : ℕ
  -- Other fields omitted for simplicity

/-- Represents the outcome of applying a strategy -/
inductive Outcome
  | IdentifiedBoth
  | NotIdentified

/-- Applies a strategy to a set of balls and returns the outcome -/
def apply_strategy (s : Strategy) (total_balls : ℕ) (radioactive_balls : ℕ) : Outcome :=
  sorry

theorem radioactive_balls_identification
  (total_balls : ℕ)
  (radioactive_balls : ℕ)
  (h_total : total_balls = 11)
  (h_radioactive : radioactive_balls = 2) :
  (∀ s : Strategy, s.num_tests < 7 → ∃ outcome, outcome = Outcome.NotIdentified) ∧
  (∃ s : Strategy, s.num_tests = 7 ∧ apply_strategy s total_balls radioactive_balls = Outcome.IdentifiedBoth) :=
sorry

end NUMINAMATH_CALUDE_radioactive_balls_identification_l2234_223454


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_inequality_l2234_223490

theorem sufficient_conditions_for_inequality (f : ℝ → ℝ) :
  (((∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x > 0)) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) ∧ (∃ x₀ : ℝ, x₀ < 0 ∧ f x₀ = 0) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_inequality_l2234_223490


namespace NUMINAMATH_CALUDE_xy_problem_l2234_223478

theorem xy_problem (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (hx : x > 0) (hy : y > 0) : y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l2234_223478


namespace NUMINAMATH_CALUDE_distance_AB_is_420_main_theorem_l2234_223424

/-- Represents a person with a speed --/
structure Person where
  speed : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  distance_AB : ℝ
  person_A : Person
  person_B : Person
  meeting_point : ℝ
  B_remaining_distance : ℝ

/-- The theorem statement --/
theorem distance_AB_is_420 (setup : ProblemSetup) : setup.distance_AB = 420 :=
  by
  have h1 : setup.person_A.speed > setup.person_B.speed := sorry
  have h2 : setup.meeting_point = setup.distance_AB - 240 := sorry
  have h3 : setup.B_remaining_distance = 120 := sorry
  have h4 : 2 * setup.person_A.speed > 2 * setup.person_B.speed := sorry
  sorry

/-- The main theorem --/
theorem main_theorem : ∃ (setup : ProblemSetup), setup.distance_AB = 420 :=
  by sorry

end NUMINAMATH_CALUDE_distance_AB_is_420_main_theorem_l2234_223424


namespace NUMINAMATH_CALUDE_max_xy_value_l2234_223473

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 16 ∧ x₀ * y₀ = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l2234_223473


namespace NUMINAMATH_CALUDE_concurrent_perpendiculars_l2234_223438

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line represented by two points -/
structure Line where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- Check if a point is inside a triangle -/
def isInside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Check if a line is perpendicular to another line -/
def isPerpendicular (l1 l2 : Line) : Prop := sorry

/-- Check if three lines are concurrent -/
def areConcurrent (l1 l2 l3 : Line) : Prop := sorry

/-- The main theorem -/
theorem concurrent_perpendiculars (innerTriangle outerTriangle : Triangle)
  (h_inside : isInside innerTriangle.A outerTriangle ∧ 
              isInside innerTriangle.B outerTriangle ∧ 
              isInside innerTriangle.C outerTriangle)
  (perpA : Line) (perpB : Line) (perpC : Line)
  (h_perpA : isPerpendicular perpA (Line.mk outerTriangle.B outerTriangle.C))
  (h_perpB : isPerpendicular perpB (Line.mk outerTriangle.C outerTriangle.A))
  (h_perpC : isPerpendicular perpC (Line.mk outerTriangle.A outerTriangle.B))
  (h_concurrent : areConcurrent perpA perpB perpC)
  (perpA' : Line) (perpB' : Line) (perpC' : Line)
  (h_perpA' : isPerpendicular perpA' (Line.mk innerTriangle.B innerTriangle.C))
  (h_perpB' : isPerpendicular perpB' (Line.mk innerTriangle.C innerTriangle.A))
  (h_perpC' : isPerpendicular perpC' (Line.mk innerTriangle.A innerTriangle.B)) :
  areConcurrent perpA' perpB' perpC' := by
    sorry


end NUMINAMATH_CALUDE_concurrent_perpendiculars_l2234_223438


namespace NUMINAMATH_CALUDE_calculation_proof_l2234_223443

theorem calculation_proof : |-4| + (1/3)⁻¹ - (Real.sqrt 2)^2 + 2035^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2234_223443


namespace NUMINAMATH_CALUDE_outfit_count_l2234_223425

/-- The number of outfits that can be made with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (green_hats red_hats blue_hats : ℕ) : ℕ :=
  red_shirts * pants * (green_hats + blue_hats) +
  green_shirts * pants * (red_hats + blue_hats) +
  blue_shirts * pants * (green_hats + red_hats)

/-- Theorem stating the number of outfits under given conditions -/
theorem outfit_count : 
  num_outfits 7 6 5 6 6 7 5 = 1284 :=
sorry

end NUMINAMATH_CALUDE_outfit_count_l2234_223425


namespace NUMINAMATH_CALUDE_max_area_at_120_l2234_223466

/-- Represents a rectangular cow pasture -/
structure Pasture where
  fence_length : ℝ
  barn_length : ℝ

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn -/
def pasture_area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fence_length - 2 * x)

/-- Theorem stating that the maximum area occurs when the side parallel to the barn is 120 feet -/
theorem max_area_at_120 (p : Pasture) (h1 : p.fence_length = 240) (h2 : p.barn_length = 500) :
  ∃ (max_x : ℝ), (∀ (x : ℝ), pasture_area p x ≤ pasture_area p max_x) ∧ p.fence_length - 2 * max_x = 120 := by
  sorry


end NUMINAMATH_CALUDE_max_area_at_120_l2234_223466


namespace NUMINAMATH_CALUDE_f_values_l2234_223451

def f (x : ℝ) : ℝ := x^2 + x + 1

theorem f_values : f 2 = 7 ∧ f (f 1) = 13 := by sorry

end NUMINAMATH_CALUDE_f_values_l2234_223451


namespace NUMINAMATH_CALUDE_payment_problem_l2234_223405

/-- The payment problem -/
theorem payment_problem (a_days b_days total_days : ℕ) (total_payment : ℚ) : 
  a_days = 6 →
  b_days = 8 →
  total_days = 3 →
  total_payment = 3680 →
  let a_work_per_day : ℚ := 1 / a_days
  let b_work_per_day : ℚ := 1 / b_days
  let ab_work_in_total_days : ℚ := (a_work_per_day + b_work_per_day) * total_days
  let c_work : ℚ := 1 - ab_work_in_total_days
  let c_payment : ℚ := c_work * total_payment
  c_payment = 460 :=
sorry

end NUMINAMATH_CALUDE_payment_problem_l2234_223405


namespace NUMINAMATH_CALUDE_siblings_total_age_l2234_223488

/-- Represents the ages of six siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ
  emily : ℕ
  david : ℕ

/-- Calculates the total age of all siblings -/
def totalAge (ages : SiblingAges) : ℕ :=
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david

/-- Theorem stating the total age of the siblings -/
theorem siblings_total_age :
  ∀ (ages : SiblingAges),
    ages.susan = 15 →
    ages.bob = 11 →
    ages.arthur = ages.susan + 2 →
    ages.tom = ages.bob - 3 →
    ages.emily = ages.susan / 2 →
    ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
    totalAge ages = 70 := by
  sorry


end NUMINAMATH_CALUDE_siblings_total_age_l2234_223488


namespace NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l2234_223497

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 - 2*Real.pi * Real.arccos (x/3) + (Real.arccos (x/3))^2 + 
  (Real.pi^2/4) * (x^2 - 9*x + 27)

theorem g_range : 
  ∀ y ∈ Set.range g, -3*(Real.pi^2/4) ≤ y ∧ y ≤ 33*(Real.pi^2/4) :=
by sorry

theorem g_range_achieves_bounds : 
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
  g x₁ = -3*(Real.pi^2/4) ∧ g x₂ = 33*(Real.pi^2/4) :=
by sorry

end NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l2234_223497


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2234_223487

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  (n ∣ (150 - 50)) ∧ 
  (n ∣ (230 - 5)) ∧ 
  (n ∣ (175 - 25)) ∧ 
  (∀ m : ℕ, m > n → (m ∣ (150 - 50)) → (m ∣ (230 - 5)) → ¬(m ∣ (175 - 25))) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2234_223487
