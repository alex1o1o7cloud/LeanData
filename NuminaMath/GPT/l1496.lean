import Mathlib

namespace NUMINAMATH_GPT_find_tan_half_sum_of_angles_l1496_149680

theorem find_tan_half_sum_of_angles (x y : ℝ) 
  (h₁ : Real.cos x + Real.cos y = 1)
  (h₂ : Real.sin x + Real.sin y = 1 / 2) : 
  Real.tan ((x + y) / 2) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_tan_half_sum_of_angles_l1496_149680


namespace NUMINAMATH_GPT_remainder_1425_1427_1429_mod_12_l1496_149678

theorem remainder_1425_1427_1429_mod_12 : 
  (1425 * 1427 * 1429) % 12 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_1425_1427_1429_mod_12_l1496_149678


namespace NUMINAMATH_GPT_exam_room_selection_l1496_149614

theorem exam_room_selection (rooms : List ℕ) (n : ℕ) 
    (fifth_room_selected : 5 ∈ rooms) (twentyfirst_room_selected : 21 ∈ rooms) :
    rooms = [5, 13, 21, 29, 37, 45, 53, 61] → 
    37 ∈ rooms ∧ 53 ∈ rooms :=
by
  sorry

end NUMINAMATH_GPT_exam_room_selection_l1496_149614


namespace NUMINAMATH_GPT_price_of_large_slice_is_250_l1496_149629

noncomputable def priceOfLargeSlice (totalSlices soldSmallSlices totalRevenue smallSlicePrice: ℕ) : ℕ :=
  let totalRevenueSmallSlices := soldSmallSlices * smallSlicePrice
  let totalRevenueLargeSlices := totalRevenue - totalRevenueSmallSlices
  let soldLargeSlices := totalSlices - soldSmallSlices
  totalRevenueLargeSlices / soldLargeSlices

theorem price_of_large_slice_is_250 :
  priceOfLargeSlice 5000 2000 1050000 150 = 250 :=
by
  sorry

end NUMINAMATH_GPT_price_of_large_slice_is_250_l1496_149629


namespace NUMINAMATH_GPT_cost_difference_is_120_l1496_149640

-- Define the monthly costs and duration
def rent_monthly_cost : ℕ := 20
def buy_monthly_cost : ℕ := 30
def months_in_a_year : ℕ := 12

-- Annual cost definitions
def annual_rent_cost : ℕ := rent_monthly_cost * months_in_a_year
def annual_buy_cost : ℕ := buy_monthly_cost * months_in_a_year

-- The main theorem to prove the difference in annual cost is $120
theorem cost_difference_is_120 : annual_buy_cost - annual_rent_cost = 120 := by
  sorry

end NUMINAMATH_GPT_cost_difference_is_120_l1496_149640


namespace NUMINAMATH_GPT_volume_of_box_l1496_149628

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_box_l1496_149628


namespace NUMINAMATH_GPT_option_D_not_equal_l1496_149693

def frac1 := (-15 : ℚ) / 12
def fracA := (-30 : ℚ) / 24
def fracB := -1 - (3 : ℚ) / 12
def fracC := -1 - (9 : ℚ) / 36
def fracD := -1 - (5 : ℚ) / 15
def fracE := -1 - (25 : ℚ) / 100

theorem option_D_not_equal :
  fracD ≠ frac1 := 
sorry

end NUMINAMATH_GPT_option_D_not_equal_l1496_149693


namespace NUMINAMATH_GPT_gift_bag_combinations_l1496_149621

theorem gift_bag_combinations (giftBags tissuePapers tags : ℕ) (h1 : giftBags = 10) (h2 : tissuePapers = 4) (h3 : tags = 5) : 
  giftBags * tissuePapers * tags = 200 := 
by 
  sorry

end NUMINAMATH_GPT_gift_bag_combinations_l1496_149621


namespace NUMINAMATH_GPT_circles_point_distance_l1496_149690

noncomputable section

-- Define the data for the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def CircleA (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := K, radius := R }

def CircleB (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := (K.1 + 2 * R, K.2), radius := R }

-- Define the condition that two circles touch each other at point K
def circles_touch (C1 C2 : Circle) (K : ℝ × ℝ) : Prop :=
  dist C1.center K = C1.radius ∧ dist C2.center K = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

-- Define the angle condition ∠AKB = 90°
def angle_AKB_is_right (A K B : ℝ × ℝ) : Prop :=
  -- Using the fact that a dot product being zero implies orthogonality
  let vec1 := (A.1 - K.1, A.2 - K.2)
  let vec2 := (B.1 - K.1, B.2 - K.2)
  vec1.1 * vec2.1 + vec1.2 * vec2.2 = 0

-- Define the points A and B being on their respective circles
def on_circle (A : ℝ × ℝ) (C : Circle) : Prop :=
  dist A C.center = C.radius

-- Define the theorem
theorem circles_point_distance 
  (R : ℝ) (K A B : ℝ × ℝ) 
  (C1 := CircleA R K) 
  (C2 := CircleB R K) 
  (h1 : circles_touch C1 C2 K) 
  (h2 : on_circle A C1) 
  (h3 : on_circle B C2) 
  (h4 : angle_AKB_is_right A K B) : 
  dist A B = 2 * R := 
sorry

end NUMINAMATH_GPT_circles_point_distance_l1496_149690


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1496_149655

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1496_149655


namespace NUMINAMATH_GPT_remainder_of_product_l1496_149633

open Nat

theorem remainder_of_product (a b : ℕ) (ha : a % 5 = 4) (hb : b % 5 = 3) :
  (a * b) % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_l1496_149633


namespace NUMINAMATH_GPT_events_mutually_exclusive_but_not_opposite_l1496_149607

inductive Card
| black
| red
| white

inductive Person
| A
| B
| C

def event_A_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.red

def event_B_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.red

theorem events_mutually_exclusive_but_not_opposite (distribution : Person → Card) :
  event_A_gets_red distribution ∧ event_B_gets_red distribution → False :=
by sorry

end NUMINAMATH_GPT_events_mutually_exclusive_but_not_opposite_l1496_149607


namespace NUMINAMATH_GPT_no_value_of_b_valid_l1496_149604

theorem no_value_of_b_valid (b n : ℤ) : b^2 + 3 * b + 1 ≠ n^2 := by
  sorry

end NUMINAMATH_GPT_no_value_of_b_valid_l1496_149604


namespace NUMINAMATH_GPT_maximum_integer_value_of_a_l1496_149648

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x - a * Real.log x

theorem maximum_integer_value_of_a (a : ℝ) (h : ∀ x ≥ 1, f x a > 0) : a ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_integer_value_of_a_l1496_149648


namespace NUMINAMATH_GPT_rectangle_breadth_l1496_149659

/-- The breadth of the rectangle is 10 units given that
1. The length of the rectangle is two-fifths of the radius of a circle.
2. The radius of the circle is equal to the side of the square.
3. The area of the square is 1225 sq. units.
4. The area of the rectangle is 140 sq. units. -/
theorem rectangle_breadth (r l b : ℝ) (h_radius : r = 35) (h_length : l = (2 / 5) * r) (h_square : 35 * 35 = 1225) (h_area_rect : l * b = 140) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_breadth_l1496_149659


namespace NUMINAMATH_GPT_books_read_by_Megan_l1496_149636

theorem books_read_by_Megan 
    (M : ℕ)
    (Kelcie : ℕ := M / 4)
    (Greg : ℕ := 2 * (M / 4) + 9)
    (total : M + Kelcie + Greg = 65) :
  M = 32 :=
by sorry

end NUMINAMATH_GPT_books_read_by_Megan_l1496_149636


namespace NUMINAMATH_GPT_eleven_pow_603_mod_500_eq_331_l1496_149646

theorem eleven_pow_603_mod_500_eq_331 : 11^603 % 500 = 331 := by
  sorry

end NUMINAMATH_GPT_eleven_pow_603_mod_500_eq_331_l1496_149646


namespace NUMINAMATH_GPT_find_value_l1496_149679

theorem find_value (N : ℝ) (h : 1.20 * N = 6000) : 0.20 * N = 1000 :=
sorry

end NUMINAMATH_GPT_find_value_l1496_149679


namespace NUMINAMATH_GPT_amount_after_3_years_l1496_149645

theorem amount_after_3_years (P t A' : ℝ) (R : ℝ) :
  P = 800 → t = 3 → A' = 992 →
  (800 * ((R + 3) / 100) * 3 = 192) →
  (A = P * (1 + (R / 100) * t)) →
  A = 1160 := by
  intros hP ht hA' hR hA
  sorry

end NUMINAMATH_GPT_amount_after_3_years_l1496_149645


namespace NUMINAMATH_GPT_second_discount_percentage_l1496_149673

theorem second_discount_percentage 
    (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (second_discount : ℝ) :
      original_price = 9795.3216374269 →
      final_price = 6700 →
      first_discount = 0.20 →
      third_discount = 0.05 →
      (original_price * (1 - first_discount) * (1 - second_discount / 100) * (1 - third_discount) = final_price) →
      second_discount = 10 :=
by
  intros h_orig h_final h_first h_third h_eq
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l1496_149673


namespace NUMINAMATH_GPT_rectangle_difference_l1496_149608

theorem rectangle_difference (A d x y : ℝ) (h1 : x * y = A) (h2 : x^2 + y^2 = d^2) :
  x - y = 2 * Real.sqrt A := 
sorry

end NUMINAMATH_GPT_rectangle_difference_l1496_149608


namespace NUMINAMATH_GPT_total_people_present_l1496_149618

/-- This definition encapsulates all the given conditions: 
    The number of parents, pupils, staff members, and performers. -/
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_staff : ℕ := 45
def num_performers : ℕ := 32

/-- Theorem stating that the total number of people present in the program is 880 
    given the stated conditions. -/
theorem total_people_present : num_parents + num_pupils + num_staff + num_performers = 880 :=
by 
  /- We can use Lean's capabilities to verify the arithmetics. -/
  sorry

end NUMINAMATH_GPT_total_people_present_l1496_149618


namespace NUMINAMATH_GPT_fair_attendance_l1496_149691

theorem fair_attendance (x y z : ℕ) 
    (h1 : y = 2 * x)
    (h2 : z = y - 200)
    (h3 : x + y + z = 2800) : x = 600 := by
  sorry

end NUMINAMATH_GPT_fair_attendance_l1496_149691


namespace NUMINAMATH_GPT_eval_expression_l1496_149642

noncomputable def T := (1 / (Real.sqrt 10 - Real.sqrt 8)) + (1 / (Real.sqrt 8 - Real.sqrt 6)) + (1 / (Real.sqrt 6 - Real.sqrt 4))

theorem eval_expression : T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1496_149642


namespace NUMINAMATH_GPT_max_value_is_one_l1496_149652

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_is_one_l1496_149652


namespace NUMINAMATH_GPT_general_formula_for_sequence_l1496_149632

noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def S_n (n : ℕ) : ℕ := sorry

theorem general_formula_for_sequence {n : ℕ} (hn: n > 0)
  (h1: ∀ n, a_n n > 0)
  (h2: ∀ n, 4 * S_n n = (a_n n)^2 + 2 * (a_n n))
  : a_n n = 2 * n := sorry

end NUMINAMATH_GPT_general_formula_for_sequence_l1496_149632


namespace NUMINAMATH_GPT_units_sold_to_customer_c_l1496_149641

theorem units_sold_to_customer_c 
  (initial_units : ℕ)
  (defective_units : ℕ)
  (units_a : ℕ)
  (units_b : ℕ)
  (units_c : ℕ)
  (h_initial : initial_units = 20)
  (h_defective : defective_units = 5)
  (h_units_a : units_a = 3)
  (h_units_b : units_b = 5)
  (h_non_defective : initial_units - defective_units = 15)
  (h_sold_all : units_a + units_b + units_c = 15) :
  units_c = 7 := by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_units_sold_to_customer_c_l1496_149641


namespace NUMINAMATH_GPT_large_diagonal_proof_l1496_149627

variable (a b : ℝ) (α : ℝ)
variable (h₁ : a < b)
variable (h₂ : 1 < a) -- arbitrary positive scalar to make obtuse properties hold

noncomputable def large_diagonal_length : ℝ :=
  Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2))

theorem large_diagonal_proof
  (h₃ : 90 < α + Real.arcsin (b * Real.sin α / a)) :
  large_diagonal_length a b α = Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2)) :=
sorry

end NUMINAMATH_GPT_large_diagonal_proof_l1496_149627


namespace NUMINAMATH_GPT_find_number_l1496_149667

theorem find_number (x : ℝ) (h : x = 12) : ( ( 17.28 / x ) / ( 3.6 * 0.2 ) ) = 2 := 
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_find_number_l1496_149667


namespace NUMINAMATH_GPT_valbonne_middle_school_l1496_149668

theorem valbonne_middle_school (students : Finset ℕ) (h : students.card = 367) :
  ∃ (date1 date2 : ℕ), date1 ≠ date2 ∧ date1 = date2 ∧ date1 ∈ students ∧ date2 ∈ students :=
by {
  sorry
}

end NUMINAMATH_GPT_valbonne_middle_school_l1496_149668


namespace NUMINAMATH_GPT_min_value_of_expression_l1496_149611

noncomputable def quadratic_function_min_value (a b c : ℝ) : ℝ :=
  (3 * (a * 1^2 + b * 1 + c) + 6 * (a * 0^2 + b * 0 + c) - (a * (-1)^2 + b * (-1) + c)) /
  ((a * 0^2 + b * 0 + c) - (a * (-2)^2 + b * (-2) + c))

theorem min_value_of_expression (a b c : ℝ)
  (h1 : b > 2 * a)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
  (h3 : a > 0) :
  quadratic_function_min_value a b c = 12 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1496_149611


namespace NUMINAMATH_GPT_two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l1496_149661

theorem two_pow_add_three_perfect_square (n : ℕ) :
  ∃ k, 2^n + 3 = k^2 ↔ n = 0 :=
by {
  sorry
}

theorem two_pow_add_one_perfect_square (n : ℕ) :
  ∃ k, 2^n + 1 = k^2 ↔ n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l1496_149661


namespace NUMINAMATH_GPT_packing_peanuts_per_large_order_l1496_149635

/-- Definitions of conditions as stated -/
def large_orders : ℕ := 3
def small_orders : ℕ := 4
def total_peanuts_used : ℕ := 800
def peanuts_per_small : ℕ := 50

/-- The statement to prove, ensuring all conditions are utilized in the definitions -/
theorem packing_peanuts_per_large_order : 
  ∃ L, large_orders * L + small_orders * peanuts_per_small = total_peanuts_used ∧ L = 200 := 
by
  use 200
  -- Adding the necessary proof steps
  have h1 : large_orders = 3 := rfl
  have h2 : small_orders = 4 := rfl
  have h3 : peanuts_per_small = 50 := rfl
  have h4 : total_peanuts_used = 800 := rfl
  sorry

end NUMINAMATH_GPT_packing_peanuts_per_large_order_l1496_149635


namespace NUMINAMATH_GPT_man_speed_3_kmph_l1496_149686

noncomputable def bullet_train_length : ℝ := 200 -- The length of the bullet train in meters
noncomputable def bullet_train_speed_kmph : ℝ := 69 -- The speed of the bullet train in km/h
noncomputable def time_to_pass_man : ℝ := 10 -- The time taken to pass the man in seconds
noncomputable def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- Conversion factor from km/h to m/s
noncomputable def bullet_train_speed_mps : ℝ := bullet_train_speed_kmph * conversion_factor_kmph_to_mps -- Speed of the bullet train in m/s
noncomputable def relative_speed : ℝ := bullet_train_length / time_to_pass_man -- Relative speed at which train passes the man
noncomputable def speed_of_man_mps : ℝ := relative_speed - bullet_train_speed_mps -- Speed of the man in m/s
noncomputable def conversion_factor_mps_to_kmph : ℝ := 3.6 -- Conversion factor from m/s to km/h
noncomputable def speed_of_man_kmph : ℝ := speed_of_man_mps * conversion_factor_mps_to_kmph -- Speed of the man in km/h

theorem man_speed_3_kmph :
  speed_of_man_kmph = 3 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_3_kmph_l1496_149686


namespace NUMINAMATH_GPT_solve_equation_l1496_149687

theorem solve_equation (x : ℝ) (h : x ≠ 1) (h_eq : x / (x - 1) = (x - 3) / (2 * x - 2)) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1496_149687


namespace NUMINAMATH_GPT_max_min_product_l1496_149688

theorem max_min_product (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x + y + z = 15) (h2 : x * y + y * z + z * x = 45) :
    ∃ m : ℝ, m = min (x * y) (min (y * z) (z * x)) ∧ m ≤ 17.5 :=
by
  sorry

end NUMINAMATH_GPT_max_min_product_l1496_149688


namespace NUMINAMATH_GPT_max_temp_difference_l1496_149654

-- Define the highest and lowest temperatures
def highest_temp : ℤ := 3
def lowest_temp : ℤ := -3

-- State the theorem for maximum temperature difference
theorem max_temp_difference : highest_temp - lowest_temp = 6 := 
by 
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_max_temp_difference_l1496_149654


namespace NUMINAMATH_GPT_Dave_guitar_strings_replacement_l1496_149672

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end NUMINAMATH_GPT_Dave_guitar_strings_replacement_l1496_149672


namespace NUMINAMATH_GPT_division_multiplication_l1496_149651

theorem division_multiplication : (0.25 / 0.005) * 2 = 100 := 
by 
  sorry

end NUMINAMATH_GPT_division_multiplication_l1496_149651


namespace NUMINAMATH_GPT_total_new_people_last_year_l1496_149624

-- Define the number of new people born and the number of people immigrated
def new_people_born : ℕ := 90171
def people_immigrated : ℕ := 16320

-- Prove that the total number of new people is 106491
theorem total_new_people_last_year : new_people_born + people_immigrated = 106491 := by
  sorry

end NUMINAMATH_GPT_total_new_people_last_year_l1496_149624


namespace NUMINAMATH_GPT_total_cost_price_l1496_149615

variables (C_table C_chair C_shelf : ℝ)

axiom h1 : 1.24 * C_table = 8091
axiom h2 : 1.18 * C_chair = 5346
axiom h3 : 1.30 * C_shelf = 11700

theorem total_cost_price :
  C_table + C_chair + C_shelf = 20055.51 :=
sorry

end NUMINAMATH_GPT_total_cost_price_l1496_149615


namespace NUMINAMATH_GPT_consequent_in_ratio_4_6_l1496_149671

theorem consequent_in_ratio_4_6 (h : 4 = 6 * (20 / x)) : x = 30 := 
by
  have h' : 4 * x = 6 * 20 := sorry -- cross-multiplication
  have h'' : x = 120 / 4 := sorry -- solving for x
  have hx : x = 30 := sorry -- simplifying 120 / 4

  exact hx

end NUMINAMATH_GPT_consequent_in_ratio_4_6_l1496_149671


namespace NUMINAMATH_GPT_inequality_proof_l1496_149666

theorem inequality_proof
  (x y : ℝ) (h1 : x^2 + x * y + y^2 = (x + y)^2 - x * y) 
  (h2 : x + y ≥ 2 * Real.sqrt (x * y)) : 
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1496_149666


namespace NUMINAMATH_GPT_andre_max_points_visited_l1496_149606
noncomputable def largest_points_to_visit_in_alphabetical_order : ℕ :=
  10

theorem andre_max_points_visited : largest_points_to_visit_in_alphabetical_order = 10 := 
by
  sorry

end NUMINAMATH_GPT_andre_max_points_visited_l1496_149606


namespace NUMINAMATH_GPT_sequence_periodicity_l1496_149657

variable {a b : ℕ → ℤ}

theorem sequence_periodicity (h : ∀ n ≥ 3, 
    (a n - a (n - 1)) * (a n - a (n - 2)) + 
    (b n - b (n - 1)) * (b n - b (n - 2)) = 0) : 
    ∃ k > 0, a k + b k = a (k + 2018) + b (k + 2018) := 
    by
    sorry

end NUMINAMATH_GPT_sequence_periodicity_l1496_149657


namespace NUMINAMATH_GPT_smallest_determinant_and_min_ab_l1496_149664

def determinant (a b : ℤ) : ℤ :=
  36 * b - 81 * a

theorem smallest_determinant_and_min_ab :
  (∃ (a b : ℤ), 0 < determinant a b ∧ determinant a b = 9 ∧ ∀ a' b', determinant a' b' = 9 → a' + b' ≥ a + b) ∧
  (∃ (a b : ℤ), a = 3 ∧ b = 7) :=
sorry

end NUMINAMATH_GPT_smallest_determinant_and_min_ab_l1496_149664


namespace NUMINAMATH_GPT_f_neg_m_l1496_149647

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the problem as a theorem
theorem f_neg_m (a b m : ℝ) (h : f a b m = 6) : f a b (-m) = -4 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_f_neg_m_l1496_149647


namespace NUMINAMATH_GPT_compound_interest_l1496_149669

noncomputable def final_amount (P : ℕ) (r : ℚ) (t : ℕ) :=
  P * ((1 : ℚ) + r) ^ t

theorem compound_interest : 
  final_amount 20000 0.20 10 = 123834.73 := 
by 
  sorry

end NUMINAMATH_GPT_compound_interest_l1496_149669


namespace NUMINAMATH_GPT_mixtilinear_incircle_radius_l1496_149682
open Real

variable (AB BC AC : ℝ)
variable (r_A : ℝ)

def triangle_conditions : Prop :=
  AB = 65 ∧ BC = 33 ∧ AC = 56

theorem mixtilinear_incircle_radius 
  (h : triangle_conditions AB BC AC)
  : r_A = 12.89 := 
sorry

end NUMINAMATH_GPT_mixtilinear_incircle_radius_l1496_149682


namespace NUMINAMATH_GPT_quadratic_function_correct_value_l1496_149656

noncomputable def quadratic_function_value (a b x x1 x2 : ℝ) :=
  a * x^2 + b * x + 5

theorem quadratic_function_correct_value
  (a b x1 x2 : ℝ)
  (h_a : a ≠ 0)
  (h_A : quadratic_function_value a b x1 x1 x2 = 2002)
  (h_B : quadratic_function_value a b x2 x1 x2 = 2002) :
  quadratic_function_value a b (x1 + x2) x1 x2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_correct_value_l1496_149656


namespace NUMINAMATH_GPT_remainder_when_divided_by_39_l1496_149617

theorem remainder_when_divided_by_39 (N : ℤ) (h1 : ∃ k : ℤ, N = 13 * k + 3) : N % 39 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_39_l1496_149617


namespace NUMINAMATH_GPT_eggs_given_by_Andrew_l1496_149623

variable (total_eggs := 222)
variable (eggs_to_buy := 67)
variable (eggs_given : ℕ)

theorem eggs_given_by_Andrew :
  eggs_given = total_eggs - eggs_to_buy ↔ eggs_given = 155 := 
by 
  sorry

end NUMINAMATH_GPT_eggs_given_by_Andrew_l1496_149623


namespace NUMINAMATH_GPT_find_principal_l1496_149696

variable (P : ℝ) (r : ℝ) (t : ℕ) (CI : ℝ) (SI : ℝ)

-- Define simple and compound interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * r * t
def compound_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r)^t - P

-- Given conditions
axiom H1 : r = 0.05
axiom H2 : t = 2
axiom H3 : compound_interest P r t - simple_interest P r t = 18

-- The principal sum is 7200
theorem find_principal : P = 7200 := 
by sorry

end NUMINAMATH_GPT_find_principal_l1496_149696


namespace NUMINAMATH_GPT_shortest_ribbon_length_is_10_l1496_149601

noncomputable def shortest_ribbon_length (L : ℕ) : Prop :=
  (∃ k1 : ℕ, L = 2 * k1) ∧ (∃ k2 : ℕ, L = 5 * k2)

theorem shortest_ribbon_length_is_10 : shortest_ribbon_length 10 :=
by
  sorry

end NUMINAMATH_GPT_shortest_ribbon_length_is_10_l1496_149601


namespace NUMINAMATH_GPT_sum_midpoints_x_sum_midpoints_y_l1496_149631

-- Defining the problem conditions
variables (a b c d e f : ℝ)
-- Sum of the x-coordinates of the triangle vertices is 15
def sum_x_coords (a b c : ℝ) : Prop := a + b + c = 15
-- Sum of the y-coordinates of the triangle vertices is 12
def sum_y_coords (d e f : ℝ) : Prop := d + e + f = 12

-- Proving the sum of x-coordinates of midpoints of sides is 15
theorem sum_midpoints_x (h1 : sum_x_coords a b c) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by  
  sorry

-- Proving the sum of y-coordinates of midpoints of sides is 12
theorem sum_midpoints_y (h2 : sum_y_coords d e f) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := 
by  
  sorry

end NUMINAMATH_GPT_sum_midpoints_x_sum_midpoints_y_l1496_149631


namespace NUMINAMATH_GPT_problem1_l1496_149684

variable {x : ℝ} {b c : ℝ}

theorem problem1 (hb : b = 9) (hc : c = -11) :
  b + c = -2 := 
by
  simp [hb, hc]
  sorry

end NUMINAMATH_GPT_problem1_l1496_149684


namespace NUMINAMATH_GPT_no_intersection_of_curves_l1496_149660

theorem no_intersection_of_curves :
  ∀ x y : ℝ, ¬ (3 * x^2 + 2 * y^2 = 4 ∧ 6 * x^2 + 3 * y^2 = 9) :=
by sorry

end NUMINAMATH_GPT_no_intersection_of_curves_l1496_149660


namespace NUMINAMATH_GPT_triangle_area_integral_bound_l1496_149689

def S := 200
def AC := 20
def dist_A_to_tangent := 25
def dist_C_to_tangent := 16
def largest_integer_not_exceeding (S : ℕ) (n : ℕ) : ℕ := n

theorem triangle_area_integral_bound (AC : ℕ) (dist_A_to_tangent : ℕ) (dist_C_to_tangent : ℕ) (S : ℕ) : 
  AC = 20 ∧ dist_A_to_tangent = 25 ∧ dist_C_to_tangent = 16 → largest_integer_not_exceeding S 20 = 10 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_integral_bound_l1496_149689


namespace NUMINAMATH_GPT_part1_part2_l1496_149697

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := 
by
  sorry

theorem part2 (h : Real.tan α = 2) : Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1496_149697


namespace NUMINAMATH_GPT_emily_jumping_game_l1496_149616

def tiles_number (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 5 = 2

theorem emily_jumping_game : tiles_number 47 :=
by
  unfold tiles_number
  sorry

end NUMINAMATH_GPT_emily_jumping_game_l1496_149616


namespace NUMINAMATH_GPT_abs_y_lt_inequality_sum_l1496_149658

-- Problem (1)
theorem abs_y_lt {
  x y : ℝ
} (h1 : |x - y| < 1) (h2 : |2 * x + y| < 1) :
  |y| < 1 := by
  sorry

-- Problem (2)
theorem inequality_sum {
  a b c d : ℝ
} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - d)) ≥ 9 / (a - d) := by
  sorry

end NUMINAMATH_GPT_abs_y_lt_inequality_sum_l1496_149658


namespace NUMINAMATH_GPT_total_dolphins_correct_l1496_149619

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end NUMINAMATH_GPT_total_dolphins_correct_l1496_149619


namespace NUMINAMATH_GPT_regular_hexagon_interior_angle_l1496_149600

theorem regular_hexagon_interior_angle : ∀ (n : ℕ), n = 6 → ∀ (angle_sum : ℕ), angle_sum = (n - 2) * 180 → (∀ (angle : ℕ), angle = angle_sum / n → angle = 120) :=
by sorry

end NUMINAMATH_GPT_regular_hexagon_interior_angle_l1496_149600


namespace NUMINAMATH_GPT_inequality_C_false_l1496_149653

theorem inequality_C_false (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : (1 / a) ^ (1 / b) ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_C_false_l1496_149653


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1496_149674

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1496_149674


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1496_149663

-- Problem 1
theorem problem1 : ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by
  sorry

-- Problem 2
theorem problem2 : ∃ n : ℕ, n = (Nat.choose 4 2) * 6 ∧ n = 36 :=
by
  sorry

-- Problem 3
theorem problem3 : ∃ n : ℕ, n = Nat.choose 4 2 ∧ n = 6 :=
by
  sorry

-- Problem 4
theorem problem4 : ∃ n : ℕ, n = 1 + (Nat.choose 4 1 + Nat.choose 4 2 / 2) + 6 ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1496_149663


namespace NUMINAMATH_GPT_Jackie_hops_six_hops_distance_l1496_149676

theorem Jackie_hops_six_hops_distance : 
  let a : ℝ := 1
  let r : ℝ := 1 / 2
  let S : ℝ := a * ((1 - r^6) / (1 - r))
  S = 63 / 32 :=
by 
  sorry

end NUMINAMATH_GPT_Jackie_hops_six_hops_distance_l1496_149676


namespace NUMINAMATH_GPT_total_cost_of_crayons_l1496_149692

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_total_cost_of_crayons_l1496_149692


namespace NUMINAMATH_GPT_photos_to_cover_poster_l1496_149644

/-
We are given a poster of dimensions 3 feet by 5 feet, and photos of dimensions 3 inches by 5 inches.
We need to prove that the number of such photos required to cover the poster is 144.
-/

-- Convert feet to inches
def feet_to_inches(feet : ℕ) : ℕ := 12 * feet

-- Dimensions of the poster in inches
def poster_height_in_inches := feet_to_inches 3
def poster_width_in_inches := feet_to_inches 5

-- Area of the poster
def poster_area : ℕ := poster_height_in_inches * poster_width_in_inches

-- Dimensions and area of one photo in inches
def photo_height := 3
def photo_width := 5
def photo_area : ℕ := photo_height * photo_width

-- Number of photos required to cover the poster
def number_of_photos : ℕ := poster_area / photo_area

-- Theorem stating the required number of photos is 144
theorem photos_to_cover_poster : number_of_photos = 144 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_photos_to_cover_poster_l1496_149644


namespace NUMINAMATH_GPT_quadratic_other_root_is_three_l1496_149610

-- Steps for creating the Lean statement following the identified conditions
variable (b : ℝ)

theorem quadratic_other_root_is_three (h1 : ∀ x : ℝ, x^2 - 2 * x - b = 0 → (x = -1 ∨ x = 3)) : 
  ∀ x : ℝ, x^2 - 2 * x - b = 0 → x = -1 ∨ x = 3 :=
by
  -- The proof is omitted
  exact h1

end NUMINAMATH_GPT_quadratic_other_root_is_three_l1496_149610


namespace NUMINAMATH_GPT_sin_bound_l1496_149670

theorem sin_bound (a : ℝ) (h : ¬ ∃ x : ℝ, Real.sin x > a) : a ≥ 1 := 
sorry

end NUMINAMATH_GPT_sin_bound_l1496_149670


namespace NUMINAMATH_GPT_factor_expr_l1496_149698

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end NUMINAMATH_GPT_factor_expr_l1496_149698


namespace NUMINAMATH_GPT_pump_rates_l1496_149677

theorem pump_rates (x y z : ℝ)
(h1 : x + y + z = 14)
(h2 : z = x + 3)
(h3 : y = 11 - 2 * x)
(h4 : 9 / x = (28 - 2 * y) / z)
: x = 3 ∧ y = 5 ∧ z = 6 :=
by
  sorry

end NUMINAMATH_GPT_pump_rates_l1496_149677


namespace NUMINAMATH_GPT_problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l1496_149609

theorem problem_85_cube_plus_3_85_square_plus_3_85_plus_1 :
  85^3 + 3 * (85^2) + 3 * 85 + 1 = 636256 := 
sorry

end NUMINAMATH_GPT_problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l1496_149609


namespace NUMINAMATH_GPT_burn_all_bridges_mod_1000_l1496_149695

theorem burn_all_bridges_mod_1000 :
  let m := 2013 * 2 ^ 2012
  let n := 3 ^ 2012
  (m + n) % 1000 = 937 :=
by
  sorry

end NUMINAMATH_GPT_burn_all_bridges_mod_1000_l1496_149695


namespace NUMINAMATH_GPT_alpha_plus_beta_l1496_149675

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem alpha_plus_beta (α β : ℝ) (hα : 0 ≤ α) (hαβ : α < Real.pi) (hβ : 0 ≤ β) (hββ : β < Real.pi)
  (hα_neq_β : α ≠ β) (hf_α : f α = 1 / 2) (hf_β : f β = 1 / 2) : α + β = (7 * Real.pi) / 6 :=
by
  sorry

end NUMINAMATH_GPT_alpha_plus_beta_l1496_149675


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_2_l1496_149625

-- Definitions
variables {a b c x : ℝ}
def quadratic_inequality_1 (a b c x : ℝ) := a * x^2 + b * x + c < 0
def quadratic_inequality_2 (a b c x : ℝ) := a * x^2 - b * x + c > 0

-- Conditions
axiom condition_1 : ∀ x, quadratic_inequality_1 a b c x ↔ (x < -2 ∨ x > -1/2)
axiom condition_2 : a < 0
axiom condition_3 : ∃ x, a * x^2 + b * x + c = 0 ∧ (x = -2 ∨ x = -1/2)
axiom condition_4 : b = 5 * a / 2
axiom condition_5 : c = a

-- Proof Problem
theorem solution_set_of_quadratic_inequality_2 : ∀ x, quadratic_inequality_2 a b c x ↔ (1/2 < x ∧ x < 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_2_l1496_149625


namespace NUMINAMATH_GPT_coin_ratio_l1496_149603

theorem coin_ratio (n₁ n₅ n₂₅ : ℕ) (total_value : ℕ) 
  (h₁ : n₁ = 40) 
  (h₅ : n₅ = 40) 
  (h₂₅ : n₂₅ = 40) 
  (hv : total_value = 70) 
  (hv_calc : n₁ * 1 + n₅ * (50 / 100) + n₂₅ * (25 / 100) = total_value) : 
  n₁ = n₅ ∧ n₁ = n₂₅ :=
by
  sorry

end NUMINAMATH_GPT_coin_ratio_l1496_149603


namespace NUMINAMATH_GPT_min_value_sin6_cos6_l1496_149620

open Real

theorem min_value_sin6_cos6 (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sin6_cos6_l1496_149620


namespace NUMINAMATH_GPT_possible_values_of_m_l1496_149650

open Set

variable (A B : Set ℤ)
variable (m : ℤ)

theorem possible_values_of_m (h₁ : A = {1, 2, m * m}) (h₂ : B = {1, m}) (h₃ : B ⊆ A) :
  m = 0 ∨ m = 2 :=
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l1496_149650


namespace NUMINAMATH_GPT_Martha_knitting_grandchildren_l1496_149649

theorem Martha_knitting_grandchildren (T_hat T_scarf T_mittens T_socks T_sweater T_total : ℕ)
  (h_hat : T_hat = 2) (h_scarf : T_scarf = 3) (h_mittens : T_mittens = 2)
  (h_socks : T_socks = 3) (h_sweater : T_sweater = 6) (h_total : T_total = 48) :
  (T_total / (T_hat + T_scarf + T_mittens + T_socks + T_sweater)) = 3 := by
  sorry

end NUMINAMATH_GPT_Martha_knitting_grandchildren_l1496_149649


namespace NUMINAMATH_GPT_zoe_total_expenditure_is_correct_l1496_149683

noncomputable def zoe_expenditure : ℝ :=
  let initial_app_cost : ℝ := 5
  let monthly_fee : ℝ := 8
  let first_two_months_fee : ℝ := 2 * monthly_fee
  let yearly_cost_without_discount : ℝ := 12 * monthly_fee
  let discount : ℝ := 0.15 * yearly_cost_without_discount
  let discounted_annual_plan : ℝ := yearly_cost_without_discount - discount
  let actual_annual_plan : ℝ := discounted_annual_plan - first_two_months_fee
  let in_game_items_cost : ℝ := 10
  let discounted_in_game_items_cost : ℝ := in_game_items_cost - (0.10 * in_game_items_cost)
  let upgraded_feature_cost : ℝ := 12
  let discounted_upgraded_feature_cost : ℝ := upgraded_feature_cost - (0.10 * upgraded_feature_cost)
  initial_app_cost + first_two_months_fee + actual_annual_plan + discounted_in_game_items_cost + discounted_upgraded_feature_cost

theorem zoe_total_expenditure_is_correct : zoe_expenditure = 122.4 :=
by
  sorry

end NUMINAMATH_GPT_zoe_total_expenditure_is_correct_l1496_149683


namespace NUMINAMATH_GPT_count_valid_n_l1496_149665

theorem count_valid_n:
  ( ∃ f: ℕ → ℕ, ∀ n, (0 < n ∧ n < 2012 → 7 ∣ (2^n - n^2) ↔ 7 ∣ (f n)) ∧ f 2012 = 576) → 
  ∃ valid_n_count: ℕ, valid_n_count = 576 := 
sorry

end NUMINAMATH_GPT_count_valid_n_l1496_149665


namespace NUMINAMATH_GPT_ratio_of_oranges_to_limes_l1496_149699

-- Constants and Definitions
def initial_fruits : ℕ := 150
def half_fruits : ℕ := 75
def oranges : ℕ := 50
def limes : ℕ := half_fruits - oranges
def ratio_oranges_limes : ℕ × ℕ := (oranges / Nat.gcd oranges limes, limes / Nat.gcd oranges limes)

-- Theorem Statement
theorem ratio_of_oranges_to_limes : ratio_oranges_limes = (2, 1) := by
  sorry

end NUMINAMATH_GPT_ratio_of_oranges_to_limes_l1496_149699


namespace NUMINAMATH_GPT_smallest_base10_integer_l1496_149685

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end NUMINAMATH_GPT_smallest_base10_integer_l1496_149685


namespace NUMINAMATH_GPT_laptop_cost_l1496_149662

theorem laptop_cost
  (C : ℝ) (down_payment := 0.2 * C + 20) (installments_paid := 65 * 4) (balance_after_4_months := 520)
  (h : C - (down_payment + installments_paid) = balance_after_4_months) :
  C = 1000 :=
by
  sorry

end NUMINAMATH_GPT_laptop_cost_l1496_149662


namespace NUMINAMATH_GPT_correct_result_l1496_149622

theorem correct_result (x : ℤ) (h : x * 3 - 5 = 103) : (x / 3) - 5 = 7 :=
sorry

end NUMINAMATH_GPT_correct_result_l1496_149622


namespace NUMINAMATH_GPT_product_of_powers_l1496_149630

theorem product_of_powers :
  ((-1 : Int)^3) * ((-2 : Int)^2) = -4 := by
  sorry

end NUMINAMATH_GPT_product_of_powers_l1496_149630


namespace NUMINAMATH_GPT_biking_distance_l1496_149602

/-- Mathematical equivalent proof problem for the distance biked -/
theorem biking_distance
  (x t d : ℕ)
  (h1 : d = (x + 1) * (3 * t / 4))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 36 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_biking_distance_l1496_149602


namespace NUMINAMATH_GPT_circle_equation_through_points_l1496_149605

-- Definitions of the points A, B, and C
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (-1, 1)

-- Prove that the equation of the circle passing through A, B, and C is (x - 1)^2 + y^2 = 5
theorem circle_equation_through_points :
  ∃ (D E F : ℝ), (∀ x y : ℝ, 
  x^2 + y^2 + D * x + E * y + F = 0 ↔
  x = -1 ∧ y = -1 ∨ 
  x = 2 ∧ y = 2 ∨ 
  x = -1 ∧ y = 1) ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x - 1)^2 + y^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_through_points_l1496_149605


namespace NUMINAMATH_GPT_equation_of_parallel_line_l1496_149613

noncomputable def is_parallel (m₁ m₂ : ℝ) := m₁ = m₂

theorem equation_of_parallel_line (m : ℝ) (b : ℝ) (x₀ y₀ : ℝ) (a b1 c : ℝ) :
  is_parallel m (1 / 2) → y₀ = -1 → x₀ = 0 → 
  (a = 1 ∧ b1 = -2 ∧ c = -2) →
  a * x₀ + b1 * y₀ + c = 0 :=
by
  intros h_parallel hy hx habc
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l1496_149613


namespace NUMINAMATH_GPT_age_twice_in_years_l1496_149637

theorem age_twice_in_years (x : ℕ) : (40 + x = 2 * (12 + x)) → x = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_twice_in_years_l1496_149637


namespace NUMINAMATH_GPT_percentage_more_than_cost_price_l1496_149638

noncomputable def SP : ℝ := 7350
noncomputable def CP : ℝ := 6681.818181818181

theorem percentage_more_than_cost_price : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_more_than_cost_price_l1496_149638


namespace NUMINAMATH_GPT_probability_green_or_blue_l1496_149639

-- Define the properties of the 10-sided die
def total_faces : ℕ := 10
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def green_faces : ℕ := 2
def blue_faces : ℕ := 1

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := green_faces + blue_faces

-- Define the probability function
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- The theorem to prove
theorem probability_green_or_blue :
  probability favorable_outcomes total_faces = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_green_or_blue_l1496_149639


namespace NUMINAMATH_GPT_blue_first_red_second_probability_l1496_149612

-- Define the initial conditions
def initial_red_marbles : ℕ := 4
def initial_white_marbles : ℕ := 6
def initial_blue_marbles : ℕ := 2
def total_marbles : ℕ := initial_red_marbles + initial_white_marbles + initial_blue_marbles

-- Probability calculation under the given conditions
def probability_blue_first : ℚ := initial_blue_marbles / total_marbles
def remaining_marbles_after_blue : ℕ := total_marbles - 1
def remaining_red_marbles : ℕ := initial_red_marbles
def probability_red_second_given_blue_first : ℚ := remaining_red_marbles / remaining_marbles_after_blue

-- Combined probability
def combined_probability : ℚ := probability_blue_first * probability_red_second_given_blue_first

-- The statement to be proved
theorem blue_first_red_second_probability :
  combined_probability = 2 / 33 :=
sorry

end NUMINAMATH_GPT_blue_first_red_second_probability_l1496_149612


namespace NUMINAMATH_GPT_shorter_side_length_l1496_149694

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 42) (h2 : a * b = 108) : b = 9 :=
by
  sorry

end NUMINAMATH_GPT_shorter_side_length_l1496_149694


namespace NUMINAMATH_GPT_sum_mod_17_l1496_149643

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_17_l1496_149643


namespace NUMINAMATH_GPT_yellow_scores_l1496_149626

theorem yellow_scores (W B : ℕ) 
  (h₁ : W / B = 7 / 6)
  (h₂ : (2 / 3 : ℚ) * (W - B) = 4) : 
  W + B = 78 :=
sorry

end NUMINAMATH_GPT_yellow_scores_l1496_149626


namespace NUMINAMATH_GPT_cats_not_liking_catnip_or_tuna_l1496_149681

theorem cats_not_liking_catnip_or_tuna :
  ∀ (total_cats catnip_lovers tuna_lovers both_lovers : ℕ),
  total_cats = 80 →
  catnip_lovers = 15 →
  tuna_lovers = 60 →
  both_lovers = 10 →
  (total_cats - (catnip_lovers - both_lovers + both_lovers + tuna_lovers - both_lovers)) = 15 :=
by
  intros total_cats catnip_lovers tuna_lovers both_lovers ht hc ht hboth
  sorry

end NUMINAMATH_GPT_cats_not_liking_catnip_or_tuna_l1496_149681


namespace NUMINAMATH_GPT_emma_age_when_sister_is_56_l1496_149634

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end NUMINAMATH_GPT_emma_age_when_sister_is_56_l1496_149634
