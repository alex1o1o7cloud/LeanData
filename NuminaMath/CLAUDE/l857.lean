import Mathlib

namespace conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l857_85731

-- Statement 1
theorem conditional_inequality_1 (a b c : ℝ) (h1 : a > b) (h2 : c ≤ 0) :
  a * c ≤ b * c := by sorry

-- Statement 2
theorem conditional_inequality_2 (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : b ≥ 0) :
  a^2 > b^2 := by sorry

-- Statement 3
theorem conditional_log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > -1) :
  Real.log (a + 1) > Real.log (b + 1) := by sorry

-- Statement 4
theorem conditional_reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : a * b > 0) :
  1 / a < 1 / b := by sorry

end conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l857_85731


namespace fraction_zero_implies_x_equals_one_l857_85789

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (x + 3) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l857_85789


namespace fraction_simplification_l857_85726

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 - 32*x^4 + 256) / (x^4 - 8) = 65 := by
sorry

end fraction_simplification_l857_85726


namespace luke_finances_duration_l857_85794

/-- Represents Luke's financial situation --/
structure LukeFinances where
  total_income : ℕ
  weekly_expenses : ℕ

/-- Calculates how many full weeks Luke's money will last --/
def weeks_money_lasts (finances : LukeFinances) : ℕ :=
  finances.total_income / finances.weekly_expenses

/-- Calculates the remaining money after the last full week --/
def remaining_money (finances : LukeFinances) : ℕ :=
  finances.total_income % finances.weekly_expenses

/-- Theorem stating how long Luke's money will last and how much will remain --/
theorem luke_finances_duration (finances : LukeFinances) 
  (h1 : finances.total_income = 34)
  (h2 : finances.weekly_expenses = 7) : 
  weeks_money_lasts finances = 4 ∧ remaining_money finances = 6 := by
  sorry

#eval weeks_money_lasts ⟨34, 7⟩
#eval remaining_money ⟨34, 7⟩

end luke_finances_duration_l857_85794


namespace complex_z_and_magnitude_l857_85730

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_z_and_magnitude : 
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - i) / (1 + i) + 2*i
  (z = i) ∧ (Complex.abs z = 1) := by
  sorry

end complex_z_and_magnitude_l857_85730


namespace election_votes_theorem_l857_85720

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : ∃ (candidate_votes : ℕ), candidate_votes = (30 * total_votes) / 100)
  (h2 : ∃ (rival_votes : ℕ), rival_votes = (70 * total_votes) / 100)
  (h3 : ∃ (candidate_votes rival_votes : ℕ), rival_votes = candidate_votes + 4000) :
  total_votes = 10000 := by
sorry

end election_votes_theorem_l857_85720


namespace min_value_product_quotient_min_value_achieved_l857_85751

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end min_value_product_quotient_min_value_achieved_l857_85751


namespace problem_statement_l857_85722

theorem problem_statement (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → 
  (40/100 : ℝ) * N = 204 := by
sorry

end problem_statement_l857_85722


namespace max_squares_visited_999_board_l857_85763

/-- A limp rook on a 999 x 999 board can move to adjacent squares and must turn at each move. -/
structure LimpRook where
  board_size : Nat
  move_to_adjacent : Bool
  must_turn : Bool

/-- A route for a limp rook is non-intersecting and cyclic. -/
structure Route where
  non_intersecting : Bool
  cyclic : Bool

/-- The maximum number of squares a limp rook can visit. -/
def max_squares_visited (rook : LimpRook) (route : Route) : Nat :=
  996000

/-- Theorem stating the maximum number of squares a limp rook can visit on a 999 x 999 board. -/
theorem max_squares_visited_999_board (rook : LimpRook) (route : Route) :
  rook.board_size = 999 ∧ rook.move_to_adjacent ∧ rook.must_turn ∧
  route.non_intersecting ∧ route.cyclic →
  max_squares_visited rook route = 996000 := by
  sorry

end max_squares_visited_999_board_l857_85763


namespace children_off_bus_l857_85795

theorem children_off_bus (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 22 → got_on = 40 → final = 2 → initial + got_on - final = 60 := by
  sorry

end children_off_bus_l857_85795


namespace remainder_theorem_l857_85762

theorem remainder_theorem (r : ℤ) : (r^15 + 1) % (r + 1) = 0 := by
  sorry

end remainder_theorem_l857_85762


namespace school_population_l857_85796

theorem school_population (G B D : ℕ) (h1 : G = 5467) (h2 : D = 1932) (h3 : B = G - D) :
  G + B = 9002 := by
  sorry

end school_population_l857_85796


namespace arithmetic_mean_of_fractions_l857_85708

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end arithmetic_mean_of_fractions_l857_85708


namespace summer_break_difference_l857_85735

theorem summer_break_difference (camp_kids : ℕ) (home_kids : ℕ) 
  (h1 : camp_kids = 819058) (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end summer_break_difference_l857_85735


namespace sqrt_2023_bound_l857_85798

theorem sqrt_2023_bound (n : ℤ) 
  (h1 : 43^2 = 1849)
  (h2 : 44^2 = 1936)
  (h3 : 45^2 = 2025)
  (h4 : 46^2 = 2116)
  (h5 : n < Real.sqrt 2023)
  (h6 : Real.sqrt 2023 < n + 1) : 
  n = 44 := by
sorry

end sqrt_2023_bound_l857_85798


namespace absolute_value_inequality_solution_set_l857_85748

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| ≤ 1} = Set.Icc (-1) 1 := by sorry

end absolute_value_inequality_solution_set_l857_85748


namespace sharon_angela_cutlery_ratio_l857_85773

/-- Prove that the ratio of Sharon's cutlery to Angela's cutlery is 2:1 -/
theorem sharon_angela_cutlery_ratio :
  let angela_pots : ℕ := 20
  let angela_plates : ℕ := 3 * angela_pots + 6
  let angela_cutlery : ℕ := angela_plates / 2
  let sharon_pots : ℕ := angela_pots / 2
  let sharon_plates : ℕ := 3 * angela_plates - 20
  let sharon_total : ℕ := 254
  let sharon_cutlery : ℕ := sharon_total - (sharon_pots + sharon_plates)
  (sharon_cutlery : ℚ) / (angela_cutlery : ℚ) = 2
  := by sorry

end sharon_angela_cutlery_ratio_l857_85773


namespace trigonometric_identity_l857_85743

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end trigonometric_identity_l857_85743


namespace only_set2_forms_triangle_l857_85749

-- Define a structure for a set of three line segments
structure TripleSegment where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality theorem
def satisfiesTriangleInequality (t : TripleSegment) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the given sets of line segments
def set1 : TripleSegment := ⟨1, 2, 3⟩
def set2 : TripleSegment := ⟨3, 4, 5⟩
def set3 : TripleSegment := ⟨4, 5, 10⟩
def set4 : TripleSegment := ⟨6, 9, 2⟩

-- State the theorem
theorem only_set2_forms_triangle :
  satisfiesTriangleInequality set2 ∧
  ¬satisfiesTriangleInequality set1 ∧
  ¬satisfiesTriangleInequality set3 ∧
  ¬satisfiesTriangleInequality set4 :=
sorry

end only_set2_forms_triangle_l857_85749


namespace acid_dilution_l857_85760

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution yields a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration := by
  sorry

#check acid_dilution

end acid_dilution_l857_85760


namespace quadratic_equation_real_roots_l857_85725

theorem quadratic_equation_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m ≥ -1 ∧ m ≠ 0) :=
sorry

end quadratic_equation_real_roots_l857_85725


namespace arithmetic_sequence_common_difference_l857_85715

theorem arithmetic_sequence_common_difference
  (a : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, a n + a (n + 1) = 4 * n)
  : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ+, a (n + 1) - a n = d :=
sorry

end arithmetic_sequence_common_difference_l857_85715


namespace cos_sin_180_degrees_l857_85738

theorem cos_sin_180_degrees :
  Real.cos (180 * π / 180) = -1 ∧ Real.sin (180 * π / 180) = 0 := by
  sorry

end cos_sin_180_degrees_l857_85738


namespace vector_dot_product_l857_85783

/-- Given two vectors a and b in ℝ² satisfying certain conditions, 
    their dot product is equal to -222/25 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + 2 * b.1 = 1 ∧ a.2 + 2 * b.2 = -3)
    (h2 : 2 * a.1 - b.1 = 1 ∧ 2 * a.2 - b.2 = 9) :
    a.1 * b.1 + a.2 * b.2 = -222 / 25 := by
  sorry

end vector_dot_product_l857_85783


namespace perfect_square_difference_l857_85754

theorem perfect_square_difference (a b c : ℕ) 
  (h1 : Nat.gcd a (Nat.gcd b c) = 1)
  (h2 : a * b = c * (a - b)) : 
  ∃ (k : ℕ), a - b = k ^ 2 := by
sorry

end perfect_square_difference_l857_85754


namespace anna_quiz_goal_impossible_l857_85758

theorem anna_quiz_goal_impossible (total_quizzes : Nat) (goal_percentage : Rat) 
  (completed_quizzes : Nat) (completed_as : Nat) : 
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  completed_as = 30 →
  ¬∃ (remaining_as : Nat), 
    (completed_as + remaining_as : Rat) / total_quizzes ≥ goal_percentage ∧ 
    remaining_as ≤ total_quizzes - completed_quizzes :=
by sorry

end anna_quiz_goal_impossible_l857_85758


namespace first_group_size_is_20_l857_85797

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the water fountain built by the first group -/
def first_fountain_length : ℝ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 7

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the water fountain built by the second group -/
def second_fountain_length : ℝ := 42

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- The theorem stating that the first group size is 20 men -/
theorem first_group_size_is_20 :
  first_group_size = 20 :=
by sorry

end first_group_size_is_20_l857_85797


namespace not_divisible_by_169_l857_85709

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end not_divisible_by_169_l857_85709


namespace book_sale_revenue_l857_85764

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  (3 * total_books = 108) →  -- Condition: 1/3 of total books is 36
  (price_per_book = 7/2) →   -- Price per book is $3.50
  (2 * total_books / 3 * price_per_book = 252) := by
  sorry

end book_sale_revenue_l857_85764


namespace grade_10_sample_size_l857_85781

/-- Represents the number of students in grade 10 -/
def grade_10_students : ℕ := sorry

/-- Represents the number of students in grade 11 -/
def grade_11_students : ℕ := grade_10_students + 300

/-- Represents the number of students in grade 12 -/
def grade_12_students : ℕ := 2 * grade_10_students

/-- The total number of students in all three grades -/
def total_students : ℕ := 3500

/-- The sampling ratio -/
def sampling_ratio : ℚ := 1 / 100

/-- Theorem stating the number of grade 10 students to be sampled -/
theorem grade_10_sample_size : 
  grade_10_students + grade_11_students + grade_12_students = total_students →
  (↑grade_10_students * sampling_ratio).floor = 8 := by
  sorry

end grade_10_sample_size_l857_85781


namespace room_length_with_veranda_l857_85718

/-- Represents a rectangular room with a surrounding veranda -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.roomLength + 2 * r.verandaWidth) * (r.roomWidth + 2 * r.verandaWidth) - r.roomLength * r.roomWidth

theorem room_length_with_veranda (r : RoomWithVeranda) :
  r.roomWidth = 12 ∧ r.verandaWidth = 2 ∧ verandaArea r = 144 → r.roomLength = 20 := by
  sorry

end room_length_with_veranda_l857_85718


namespace arithmetic_sequence_squares_l857_85713

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ a d : ℚ, (36 + k : ℚ) = (a - d)^2 ∧ 
               (300 + k : ℚ) = a^2 ∧ 
               (596 + k : ℚ) = (a + d)^2) ↔ 
  k = 925 := by
sorry

end arithmetic_sequence_squares_l857_85713


namespace vector_simplification_l857_85727

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O P Q M : V)

-- State the theorem
theorem vector_simplification :
  (P - O) + (Q - P) - (Q - M) = M - O :=
by sorry

end vector_simplification_l857_85727


namespace binomial_coefficient_inequality_l857_85772

theorem binomial_coefficient_inequality
  (n k h : ℕ)
  (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
by sorry

end binomial_coefficient_inequality_l857_85772


namespace min_max_sum_l857_85777

theorem min_max_sum (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  max (a + b + c) (max (b + c + d) (max (c + d + e) (max (d + e + f) (e + f + g)))) ≥ 1/3 := by
  sorry

end min_max_sum_l857_85777


namespace triangle_area_l857_85782

theorem triangle_area (a b c : ℝ) (A : ℝ) : 
  b = 3 → 
  a - c = 2 → 
  A = 2 * Real.pi / 3 → 
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 :=
by sorry

end triangle_area_l857_85782


namespace rectangle_area_l857_85766

def circle_inscribed_rectangle (r : ℝ) (l w : ℝ) : Prop :=
  2 * r = w

def length_width_ratio (l w : ℝ) : Prop :=
  l = 3 * w

theorem rectangle_area (r l w : ℝ) 
  (h1 : circle_inscribed_rectangle r l w) 
  (h2 : length_width_ratio l w) 
  (h3 : r = 7) : l * w = 588 := by
  sorry

end rectangle_area_l857_85766


namespace ipod_original_price_l857_85706

theorem ipod_original_price (discount_percent : ℝ) (final_price : ℝ) (original_price : ℝ) : 
  discount_percent = 35 →
  final_price = 83.2 →
  final_price = original_price * (1 - discount_percent / 100) →
  original_price = 128 := by
sorry

end ipod_original_price_l857_85706


namespace four_door_room_ways_l857_85744

/-- The number of ways to enter or exit a room with a given number of doors. -/
def waysToEnterOrExit (numDoors : ℕ) : ℕ := numDoors

/-- The number of different ways to enter and exit a room with a given number of doors. -/
def totalWays (numDoors : ℕ) : ℕ :=
  (waysToEnterOrExit numDoors) * (waysToEnterOrExit numDoors)

/-- Theorem: In a room with four doors, there are 16 different ways to enter and exit. -/
theorem four_door_room_ways :
  totalWays 4 = 16 := by
  sorry

end four_door_room_ways_l857_85744


namespace complex_number_problem_l857_85775

theorem complex_number_problem (a b c : ℂ) 
  (h_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 8) :
  a = 3 := by
  sorry

end complex_number_problem_l857_85775


namespace cylindrical_to_rectangular_conversion_l857_85729

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (hr : r = 7) 
  (hθ : θ = π / 3) 
  (hz : z = -3) :
  ∃ (x y : ℝ), 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    x = 3.5 ∧ 
    y = 7 * Real.sqrt 3 / 2 ∧ 
    z = -3 := by
  sorry

end cylindrical_to_rectangular_conversion_l857_85729


namespace unique_digit_solution_l857_85780

theorem unique_digit_solution :
  ∃! (E U L S R T : ℕ),
    (E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (L ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    (T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧
    E ≠ U ∧ E ≠ L ∧ E ≠ S ∧ E ≠ R ∧ E ≠ T ∧
    U ≠ L ∧ U ≠ S ∧ U ≠ R ∧ U ≠ T ∧
    L ≠ S ∧ L ≠ R ∧ L ≠ T ∧
    S ≠ R ∧ S ≠ T ∧
    R ≠ T ∧
    E + U + L = 6 ∧
    S + R + U + T = 18 ∧
    U * T = 15 ∧
    S * L = 8 ∧
    E = 1 ∧ U = 3 ∧ L = 2 ∧ S = 4 ∧ R = 6 ∧ T = 5 :=
by sorry

end unique_digit_solution_l857_85780


namespace abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l857_85721

theorem abs_m_eq_abs_half_implies_m_eq_plus_minus_half (m : ℝ) : 
  |(-m)| = |(-1/2)| → m = -1/2 ∨ m = 1/2 := by
sorry

end abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l857_85721


namespace mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l857_85746

-- Define the cost of items
def bagel_cost : ℝ := 1.5
def coffee_cost : ℝ := 3.25

-- Define Mrs. Martin's purchase
def mrs_martin_coffee : ℕ := 3
def mrs_martin_bagels : ℕ := 2
def mrs_martin_total : ℝ := 12.75

-- Define Mr. Martin's purchase
def mr_martin_bagels : ℕ := 5
def mr_martin_total : ℝ := 14.00

-- Theorem to prove
theorem mr_martin_coffee_cups : ℕ := by
  -- The number of coffee cups Mr. Martin bought
  sorry

-- Verify Mrs. Martin's purchase
theorem verify_mrs_martin_purchase :
  mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost = mrs_martin_total := by
  sorry

-- Verify Mr. Martin's purchase
theorem verify_mr_martin_purchase :
  mr_martin_coffee_cups * coffee_cost + mr_martin_bagels * bagel_cost = mr_martin_total := by
  sorry

end mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l857_85746


namespace no_solution_iff_m_equals_one_l857_85785

theorem no_solution_iff_m_equals_one :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 3 → ((3 - 2*x) / (x - 3) - (m*x - 2) / (3 - x) ≠ -1)) ↔ m = 1 := by
  sorry

end no_solution_iff_m_equals_one_l857_85785


namespace staircase_shape_perimeter_l857_85707

/-- Represents the shape described in the problem -/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  staircase_sides : ℕ
  area : ℝ

/-- Calculates the perimeter of the StaircaseShape -/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + 4 + 5 + (shape.staircase_sides : ℝ)

/-- Theorem stating the perimeter of the specific shape described in the problem -/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 12 ∧ 
    shape.staircase_sides = 10 ∧ 
    shape.area = 72 → 
    perimeter shape = 42.25 := by
  sorry


end staircase_shape_perimeter_l857_85707


namespace wilson_class_blue_eyes_l857_85742

/-- Represents the class composition -/
structure ClassComposition where
  total : ℕ
  blond_to_blue_ratio : Rat
  both_traits : ℕ
  neither_trait : ℕ

/-- Calculates the number of blue-eyed students -/
def blue_eyed_count (c : ClassComposition) : ℕ :=
  sorry

/-- Theorem stating the number of blue-eyed students in Mrs. Wilson's class -/
theorem wilson_class_blue_eyes :
  let c : ClassComposition := {
    total := 40,
    blond_to_blue_ratio := 3 / 2,
    both_traits := 8,
    neither_trait := 5
  }
  blue_eyed_count c = 18 := by sorry

end wilson_class_blue_eyes_l857_85742


namespace amaro_roses_l857_85702

theorem amaro_roses :
  ∀ (total_roses : ℕ),
  (3 * total_roses / 4 : ℚ) + (3 * total_roses / 16 : ℚ) = 75 →
  total_roses = 80 := by
sorry

end amaro_roses_l857_85702


namespace complete_square_constant_l857_85716

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end complete_square_constant_l857_85716


namespace at_most_one_acute_forming_point_l857_85734

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function to check if a triangle is acute-angled -/
def isAcuteTriangle (p q r : Point) : Prop :=
  sorry -- Definition of acute triangle

/-- The theorem stating that at most one point can form acute triangles with any other two points -/
theorem at_most_one_acute_forming_point (points : Finset Point) (h : points.card = 2006) :
  ∃ (p : Point), p ∈ points ∧
    (∀ (q r : Point), q ∈ points → r ∈ points → q ≠ r → q ≠ p → r ≠ p → isAcuteTriangle p q r) →
    ∀ (p' : Point), p' ∈ points → p' ≠ p →
      ∃ (q r : Point), q ∈ points ∧ r ∈ points ∧ q ≠ r ∧ q ≠ p' ∧ r ≠ p' ∧ ¬isAcuteTriangle p' q r :=
by
  sorry

end at_most_one_acute_forming_point_l857_85734


namespace star_to_square_ratio_is_three_fifths_l857_85770

/-- Represents a square with side length 5 cm containing a star formed by four identical isosceles triangles, each with height 1 cm -/
structure StarInSquare where
  square_side : ℝ
  triangle_height : ℝ
  square_side_eq : square_side = 5
  triangle_height_eq : triangle_height = 1

/-- Calculates the ratio of the star area to the square area -/
def star_to_square_ratio (s : StarInSquare) : ℚ :=
  3 / 5

/-- Theorem stating that the ratio of the star area to the square area is 3/5 -/
theorem star_to_square_ratio_is_three_fifths (s : StarInSquare) :
  star_to_square_ratio s = 3 / 5 := by
  sorry

end star_to_square_ratio_is_three_fifths_l857_85770


namespace football_tournament_semifinal_probability_l857_85745

theorem football_tournament_semifinal_probability :
  let num_teams : ℕ := 8
  let num_semifinal_pairs : ℕ := 2
  let prob_win_match : ℚ := 1 / 2
  
  -- Probability of team B being in the correct subgroup
  let prob_correct_subgroup : ℚ := num_semifinal_pairs / (num_teams - 1)
  
  -- Probability of both teams winning their matches to reach semifinals
  let prob_both_win : ℚ := prob_win_match * prob_win_match
  
  -- Total probability
  prob_correct_subgroup * prob_both_win = 1 / 14 :=
by sorry

end football_tournament_semifinal_probability_l857_85745


namespace inequality_proof_l857_85788

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1/2 := by
  sorry

end inequality_proof_l857_85788


namespace prob_three_draws_equals_36_125_l857_85793

/-- The probability of drawing exactly 3 balls to get two red balls -/
def prob_three_draws (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  2 * (p_red * p_white * p_red)

/-- The box contains 3 red balls and 2 white balls -/
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

theorem prob_three_draws_equals_36_125 :
  prob_three_draws total_balls red_balls white_balls = 36 / 125 := by
  sorry

end prob_three_draws_equals_36_125_l857_85793


namespace apple_cost_price_l857_85700

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 16 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - loss_fraction * cost_price ∧ 
    cost_price = 19.2 := by
  sorry

end apple_cost_price_l857_85700


namespace x_range_for_inequality_l857_85759

theorem x_range_for_inequality (x t : ℝ) :
  (t ∈ Set.Icc 1 3) →
  (((1/8) * (2*x - x^2) ≤ t^2 - 3*t + 2) ∧ (t^2 - 3*t + 2 ≤ 3 - x^2)) →
  (x ∈ Set.Icc (-1) (1 - Real.sqrt 3)) := by
sorry

end x_range_for_inequality_l857_85759


namespace missing_number_proof_l857_85728

theorem missing_number_proof (x : ℤ) : (4 + 3) + (8 - x - 1) = 11 → x = 3 := by
  sorry

end missing_number_proof_l857_85728


namespace expansion_coefficient_l857_85714

/-- The coefficient of x^4 in the expansion of (1 + √x)^10 -/
def coefficient_x4 : ℕ :=
  Nat.choose 10 8

theorem expansion_coefficient (n : ℕ) :
  coefficient_x4 = 45 := by
  sorry

end expansion_coefficient_l857_85714


namespace geometric_sequence_first_term_l857_85752

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} satisfying a_1 + a_6 = 11 and a_3 * a_4 = 32/9,
    prove that a_1 = 32/3 or a_1 = 1/3 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_sum : a 1 + a 6 = 11)
  (h_prod : a 3 * a 4 = 32/9) :
  a 1 = 32/3 ∨ a 1 = 1/3 :=
sorry

end geometric_sequence_first_term_l857_85752


namespace six_digit_number_representation_l857_85765

theorem six_digit_number_representation (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (1000 ≤ b ∧ b < 10000) →  -- b is a four-digit number
  (100000 ≤ 10000 * a + b ∧ 10000 * a + b < 1000000) →  -- result is a six-digit number
  10000 * a + b = 10000 * a + b :=  -- the representation is correct
by sorry

end six_digit_number_representation_l857_85765


namespace election_votes_l857_85784

theorem election_votes (total_votes : ℕ) (winner_votes : ℕ) 
  (diff1 diff2 diff3 : ℕ) : 
  total_votes = 963 →
  winner_votes - diff1 + winner_votes - diff2 + winner_votes - diff3 + winner_votes = total_votes →
  diff1 = 53 →
  diff2 = 79 →
  diff3 = 105 →
  winner_votes = 300 :=
by sorry

end election_votes_l857_85784


namespace system_solution_existence_and_values_l857_85753

/-- Given a system of equations with parameters α₁, α₂, α₃, α₄, prove that a solution exists
    if and only if α₁ = α₂ = α₃ = α or α₄ = α, and find the solution in this case. -/
theorem system_solution_existence_and_values (α₁ α₂ α₃ α₄ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ + x₂ = α₁ * α₂ ∧
    x₁ + x₃ = α₁ * α₃ ∧
    x₁ + x₄ = α₁ * α₄ ∧
    x₂ + x₃ = α₂ * α₃ ∧
    x₂ + x₄ = α₂ * α₄ ∧
    x₃ + x₄ = α₃ * α₄) ↔
  ((α₁ = α₂ ∧ α₂ = α₃) ∨ α₄ = α₂) ∧
  (∃ α β : ℝ,
    (α = α₁ ∧ β = α₄) ∨ (α = α₂ ∧ β = α₁) ∧
    x₁ = α^2 / 2 ∧
    x₂ = α^2 / 2 ∧
    x₃ = α^2 / 2 ∧
    x₄ = α * (β - α / 2)) :=
by sorry


end system_solution_existence_and_values_l857_85753


namespace min_product_with_constraint_min_product_achievable_l857_85750

theorem min_product_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 20 * a * b = 13 * a + 14 * b) : a * b ≥ 1.82 := by
  sorry

theorem min_product_achievable : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  20 * a * b = 13 * a + 14 * b ∧ a * b = 1.82 := by
  sorry

end min_product_with_constraint_min_product_achievable_l857_85750


namespace office_age_problem_l857_85717

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) (group1_persons : ℕ) (avg_age_group1 : ℝ) (group2_persons : ℕ) (person15_age : ℝ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_persons = 5 →
  avg_age_group1 = 14 →
  group2_persons = 9 →
  person15_age = 56 →
  (total_persons * avg_age_all - group1_persons * avg_age_group1 - person15_age) / group2_persons = 16 := by
  sorry

end office_age_problem_l857_85717


namespace tan_value_from_trig_equation_l857_85756

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end tan_value_from_trig_equation_l857_85756


namespace fraction_numerator_l857_85790

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : x / y * y + 3 * y / 10 = 1 / 2 * y) : x = 1 / 5 := by
  sorry

end fraction_numerator_l857_85790


namespace trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l857_85771

-- Part I
theorem trig_sum_equals_negative_sqrt3_over_6 :
  Real.sin (5 * Real.pi / 3) + Real.cos (11 * Real.pi / 2) + Real.tan (-11 * Real.pi / 6) = -Real.sqrt 3 / 6 := by
  sorry

-- Part II
theorem trig_fraction_sum_simplification (θ : Real) 
  (h1 : Real.tan θ ≠ 0) (h2 : Real.tan θ ≠ 1) :
  (Real.sin θ / (1 - 1 / Real.tan θ)) + (Real.cos θ / (1 - Real.tan θ)) = Real.sin θ + Real.cos θ := by
  sorry

end trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l857_85771


namespace problem_solution_l857_85791

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) : 
  m = 4 ∧ a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 + c₀^2 = Real.sqrt 3 ∧ 
  4 * a₀^4 + 4 * b₀^4 + 4 * c₀^4 = m := by
  sorry

end problem_solution_l857_85791


namespace perpendicular_vectors_x_value_l857_85769

/-- Given two vectors a and b in ℝ², where a = (-5, 1) and b = (2, x),
    if a and b are perpendicular, then x = 10. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-5, 1)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 10 := by
sorry

end perpendicular_vectors_x_value_l857_85769


namespace divisors_of_27n_cubed_l857_85703

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_27n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : num_divisors n = 12) :
  num_divisors (27 * n^3) = 256 := by sorry

end divisors_of_27n_cubed_l857_85703


namespace range_of_f_is_real_l857_85761

noncomputable def f (x : ℝ) := x^3 - 3*x

theorem range_of_f_is_real : 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (f 2 = 2) ∧ 
  (deriv f 2 = 9) :=
by sorry

end range_of_f_is_real_l857_85761


namespace quadratic_form_h_value_l857_85747

theorem quadratic_form_h_value (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 15 = a * (x + 3/2)^2 + k :=
by
  sorry

end quadratic_form_h_value_l857_85747


namespace bicycle_cost_calculation_l857_85739

/-- Given two bicycles sold at a certain price, with specified profit and loss percentages,
    calculate the total cost of both bicycles. -/
theorem bicycle_cost_calculation 
  (selling_price : ℚ) 
  (profit_percent : ℚ) 
  (loss_percent : ℚ) : 
  selling_price = 990 →
  profit_percent = 10 / 100 →
  loss_percent = 10 / 100 →
  ∃ (cost1 cost2 : ℚ),
    cost1 * (1 + profit_percent) = selling_price ∧
    cost2 * (1 - loss_percent) = selling_price ∧
    cost1 + cost2 = 2000 := by
  sorry

end bicycle_cost_calculation_l857_85739


namespace bisection_exact_solution_possible_l857_85732

/-- The bisection method can potentially find an exact solution -/
theorem bisection_exact_solution_possible
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : Continuous f) (hfab : f a * f b < 0) :
  ∃ x ∈ Set.Icc a b, f x = 0 ∧ ∃ n : ℕ, x = (a + b) / 2^(n + 1) :=
sorry

end bisection_exact_solution_possible_l857_85732


namespace raduzhny_population_is_900_l857_85787

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe -/
def znoynoe_population : ℕ := 1000

/-- The difference between Znoynoe's population and the average village population -/
def population_difference : ℕ := 90

/-- The maximum population difference between any village and Znoynoe -/
def max_population_difference : ℕ := 100

/-- The total population of all villages except Znoynoe -/
def other_villages_population : ℕ := (num_villages - 1) * (znoynoe_population - population_difference)

/-- The population of Raduzhny -/
def raduzhny_population : ℕ := other_villages_population / (num_villages - 1)

theorem raduzhny_population_is_900 :
  raduzhny_population = 900 :=
sorry

end raduzhny_population_is_900_l857_85787


namespace river_width_l857_85733

/-- A configuration of points for measuring river width -/
structure RiverMeasurement where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  AC_eq_40 : dist A C = 40
  CD_eq_12 : dist C D = 12
  AE_eq_24 : dist A E = 24
  EC_eq_16 : dist E C = 16
  AB_perp_CD : ((B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) : ℝ) = 0
  E_on_AB : ∃ t : ℝ, E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The width of the river is 18 meters -/
theorem river_width (m : RiverMeasurement) : dist m.A m.B = 18 := by
  sorry

end river_width_l857_85733


namespace pot_height_problem_shorter_pot_height_l857_85786

theorem pot_height_problem (h₁ b₁ b₂ : ℝ) (h₁_pos : 0 < h₁) (b₁_pos : 0 < b₁) (b₂_pos : 0 < b₂) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem shorter_pot_height (tall_pot_height tall_pot_shadow short_pot_shadow : ℝ)
  (tall_pot_height_pos : 0 < tall_pot_height)
  (tall_pot_shadow_pos : 0 < tall_pot_shadow)
  (short_pot_shadow_pos : 0 < short_pot_shadow)
  (h_tall : tall_pot_height = 40)
  (h_tall_shadow : tall_pot_shadow = 20)
  (h_short_shadow : short_pot_shadow = 10) :
  tall_pot_height * short_pot_shadow / tall_pot_shadow = 20 :=
by sorry

end pot_height_problem_shorter_pot_height_l857_85786


namespace star_operation_sum_l857_85774

theorem star_operation_sum (c d : ℕ) : 
  c ≥ 2 → d ≥ 2 → c^d + c*d = 42 → c + d = 7 := by
  sorry

end star_operation_sum_l857_85774


namespace bulls_win_in_seven_l857_85719

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Bulls winning a single game -/
def p_bulls_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The total number of games played when the series goes to 7 games -/
def total_games : ℕ := 7

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

theorem bulls_win_in_seven (
  p_knicks_win : ℚ) 
  (p_bulls_win : ℚ) 
  (games_to_win : ℕ) 
  (total_games : ℕ) 
  (ways_to_choose_3_of_6 : ℕ) :
  p_knicks_win = 3/4 →
  p_bulls_win = 1 - p_knicks_win →
  games_to_win = 4 →
  total_games = 7 →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_bulls_win^3 * p_knicks_win^3 * p_bulls_win = 540/16384 :=
by sorry

end bulls_win_in_seven_l857_85719


namespace correct_product_l857_85799

theorem correct_product (a b c : ℚ) (h1 : a = 0.005) (h2 : b = 3.24) (h3 : c = 0.0162) 
  (h4 : (5 : ℚ) * 324 = 1620) : a * b = c := by
  sorry

end correct_product_l857_85799


namespace table_rearrangement_l857_85741

/-- Represents a table with n rows and n columns -/
def Table (α : Type) (n : ℕ) := Fin n → Fin n → α

/-- Predicate to check if a row has no repeated elements -/
def NoRepeatsInRow {α : Type} [DecidableEq α] (row : Fin n → α) : Prop :=
  ∀ i j : Fin n, i ≠ j → row i ≠ row j

/-- Predicate to check if a table has no repeated elements in any row -/
def NoRepeatsInRows {α : Type} [DecidableEq α] (T : Table α n) : Prop :=
  ∀ i : Fin n, NoRepeatsInRow (T i)

/-- Predicate to check if two rows are permutations of each other -/
def RowsArePermutations {α : Type} [DecidableEq α] (row1 row2 : Fin n → α) : Prop :=
  ∀ x : α, (∃ i : Fin n, row1 i = x) ↔ (∃ j : Fin n, row2 j = x)

/-- Predicate to check if a column has no repeated elements -/
def NoRepeatsInColumn {α : Type} [DecidableEq α] (T : Table α n) (j : Fin n) : Prop :=
  ∀ i k : Fin n, i ≠ k → T i j ≠ T k j

/-- The main theorem statement -/
theorem table_rearrangement {α : Type} [DecidableEq α] (n : ℕ) (T : Table α n) 
  (h : NoRepeatsInRows T) :
  ∃ T_star : Table α n,
    (∀ i : Fin n, RowsArePermutations (T i) (T_star i)) ∧
    (∀ j : Fin n, NoRepeatsInColumn T_star j) :=
  sorry

end table_rearrangement_l857_85741


namespace negative_option_l857_85792

theorem negative_option : ∃ (x : ℝ), x < 0 ∧ 
  x = -(-5)^2 ∧ 
  -(-5) ≥ 0 ∧ 
  |-5| ≥ 0 ∧ 
  (-5) * (-5) ≥ 0 :=
by sorry

end negative_option_l857_85792


namespace triangle_shape_from_complex_product_l857_85779

open Complex

/-- Given a triangle ABC with sides a, b and angles A, B, C,
    if z₁ = a + bi and z₂ = cos A + i cos B, and their product is purely imaginary,
    then the triangle is either isosceles or right-angled. -/
theorem triangle_shape_from_complex_product (a b : ℝ) (A B C : ℝ) :
  let z₁ : ℂ := ⟨a, b⟩
  let z₂ : ℂ := ⟨Real.cos A, Real.cos B⟩
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →  -- Triangle conditions
  (z₁ * z₂).re = 0 →  -- Product is purely imaginary
  (A = B) ∨ (A + B = π / 2) :=  -- Triangle is isosceles or right-angled
by sorry

end triangle_shape_from_complex_product_l857_85779


namespace complement_intersection_theorem_l857_85737

universe u

def U : Set ℤ := {-1, 0, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end complement_intersection_theorem_l857_85737


namespace subtraction_preserves_inequality_l857_85776

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l857_85776


namespace widgets_per_carton_is_three_l857_85736

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the number of widgets per carton -/
def widgetsPerCarton (cartonDim : BoxDimensions) (shippingBoxDim : BoxDimensions) (totalWidgets : ℕ) : ℕ :=
  let cartonsPerLayer := (shippingBoxDim.width / cartonDim.width) * (shippingBoxDim.length / cartonDim.length)
  let layers := shippingBoxDim.height / cartonDim.height
  let totalCartons := cartonsPerLayer * layers
  totalWidgets / totalCartons

theorem widgets_per_carton_is_three :
  let cartonDim : BoxDimensions := ⟨4, 4, 5⟩
  let shippingBoxDim : BoxDimensions := ⟨20, 20, 20⟩
  let totalWidgets : ℕ := 300
  widgetsPerCarton cartonDim shippingBoxDim totalWidgets = 3 := by
  sorry

end widgets_per_carton_is_three_l857_85736


namespace sum_of_zero_seven_representable_l857_85767

/-- A function that checks if a real number can be written using only 0 and 7 in decimal notation -/
def uses_only_zero_and_seven (x : ℝ) : Prop :=
  ∃ (digits : ℕ → ℕ), (∀ n, digits n ∈ ({0, 7} : Set ℕ)) ∧
    x = ∑' n, (digits n : ℝ) / 10^n

/-- Theorem stating that any positive real number can be represented as the sum of nine numbers,
    each of which in decimal notation consists of the digits 0 and 7 -/
theorem sum_of_zero_seven_representable (x : ℝ) (hx : 0 < x) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ),
    x = a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ ∧
    (uses_only_zero_and_seven a₁) ∧
    (uses_only_zero_and_seven a₂) ∧
    (uses_only_zero_and_seven a₃) ∧
    (uses_only_zero_and_seven a₄) ∧
    (uses_only_zero_and_seven a₅) ∧
    (uses_only_zero_and_seven a₆) ∧
    (uses_only_zero_and_seven a₇) ∧
    (uses_only_zero_and_seven a₈) ∧
    (uses_only_zero_and_seven a₉) :=
by
  sorry

end sum_of_zero_seven_representable_l857_85767


namespace cylinder_surface_area_l857_85723

theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (hπ : h = 6 * Real.pi) (wπ : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end cylinder_surface_area_l857_85723


namespace jerry_collection_cost_l857_85757

/-- The amount of money Jerry needs to complete his action figure collection -/
def jerry_needs_money (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to complete his collection -/
theorem jerry_collection_cost : jerry_needs_money 7 16 8 = 72 := by
  sorry

end jerry_collection_cost_l857_85757


namespace farmers_extra_days_l857_85710

/-- A farmer's ploughing problem -/
theorem farmers_extra_days
  (total_area : ℕ)
  (planned_daily_area : ℕ)
  (actual_daily_area : ℕ)
  (area_left : ℕ)
  (h1 : total_area = 720)
  (h2 : planned_daily_area = 120)
  (h3 : actual_daily_area = 85)
  (h4 : area_left = 40) :
  ∃ extra_days : ℕ,
    actual_daily_area * (total_area / planned_daily_area + extra_days) = total_area - area_left ∧
    extra_days = 2 := by
  sorry

#check farmers_extra_days

end farmers_extra_days_l857_85710


namespace min_value_of_f_l857_85755

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: x = 3 minimizes the function f(x) = 3x^2 - 18x + 7 -/
theorem min_value_of_f :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

end min_value_of_f_l857_85755


namespace custom_op_solution_l857_85724

/-- The custom operation ※ -/
def custom_op (a b : ℕ) : ℕ := (b * (2 * a + b - 1)) / 2

theorem custom_op_solution :
  ∀ a : ℕ, custom_op a 15 = 165 → a = 4 :=
by
  sorry

end custom_op_solution_l857_85724


namespace arrangements_count_is_correct_l857_85778

/-- The number of ways to divide 2 teachers and 4 students into two groups,
    each containing 1 teacher and 2 students, and then assign these groups to two locations -/
def arrangementsCount : ℕ := 12

/-- The number of ways to choose 2 students from 4 students -/
def waysToChooseStudents : ℕ := Nat.choose 4 2

/-- The number of ways to choose 2 students from 2 students (always 1) -/
def waysToChooseRemainingStudents : ℕ := Nat.choose 2 2

/-- The number of ways to assign 2 groups to 2 locations -/
def waysToAssignGroups : ℕ := 2

theorem arrangements_count_is_correct :
  arrangementsCount = waysToChooseStudents * waysToChooseRemainingStudents * waysToAssignGroups :=
by sorry

end arrangements_count_is_correct_l857_85778


namespace blue_paint_calculation_l857_85711

/-- Given a ratio of blue to white paint and an amount of white paint, 
    calculate the amount of blue paint required. -/
def blue_paint_amount (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the blue paint amount is 12 quarts. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 4
  let white_ratio : ℚ := 5
  let white_amount : ℚ := 15
  blue_paint_amount blue_ratio white_ratio white_amount = 12 := by
sorry

#eval blue_paint_amount 4 5 15

end blue_paint_calculation_l857_85711


namespace line_circle_separation_l857_85712

/-- If a point (a,b) is inside the unit circle, then the line ax + by = 1 is separated from the circle -/
theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∃ (d : ℝ), d > 1 ∧ d = 1 / Real.sqrt (a^2 + b^2) := by
  sorry

end line_circle_separation_l857_85712


namespace quadratic_coefficient_l857_85705

/-- Quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ :=
  evaluate f (vertex_x f)

theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = 5)
  (h3 : evaluate f 3 = 4) :
  f.a = -1 := by
  sorry

end quadratic_coefficient_l857_85705


namespace sqrt_900_squared_times_6_l857_85701

theorem sqrt_900_squared_times_6 : (Real.sqrt 900)^2 * 6 = 5400 := by
  sorry

end sqrt_900_squared_times_6_l857_85701


namespace rhombus_perimeter_l857_85768

theorem rhombus_perimeter (d : ℝ) (h1 : d = 20) : 
  let longer_diagonal := 1.3 * d
  let side := Real.sqrt ((d/2)^2 + (longer_diagonal/2)^2)
  4 * side = 4 * Real.sqrt 269 := by sorry

end rhombus_perimeter_l857_85768


namespace range_of_m_l857_85704

def f (x : ℝ) : ℝ := x^2 - 4*x - 2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end range_of_m_l857_85704


namespace complex_equation_solution_l857_85740

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 2 → z = 3 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l857_85740
