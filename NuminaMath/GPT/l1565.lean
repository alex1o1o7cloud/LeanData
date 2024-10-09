import Mathlib

namespace katya_needs_at_least_ten_l1565_156589

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l1565_156589


namespace triangle_DFG_area_l1565_156587

theorem triangle_DFG_area (a b x y : ℝ) (h_ab : a * b = 20) (h_xy : x * y = 8) : 
  (a * b - x * y) / 2 = 6 := 
by
  sorry

end triangle_DFG_area_l1565_156587


namespace grades_with_fewer_students_l1565_156595

-- Definitions of the involved quantities
variables (G1 G2 G5 G1_2 : ℕ)
variables (Set_X : ℕ)

-- Conditions given in the problem
theorem grades_with_fewer_students (h1: G1_2 = Set_X + 30) (h2: G5 = G1 - 30) :
  exists Set_X, G1_2 - Set_X = 30 :=
by 
  sorry

end grades_with_fewer_students_l1565_156595


namespace fraction_spent_on_DVDs_l1565_156504

theorem fraction_spent_on_DVDs (initial_money spent_on_books additional_books_cost remaining_money_spent fraction remaining_money_after_DVDs : ℚ) : 
  initial_money = 320 ∧
  spent_on_books = initial_money / 4 ∧
  additional_books_cost = 10 ∧
  remaining_money_spent = 230 ∧
  remaining_money_after_DVDs = 130 ∧
  remaining_money_spent = initial_money - (spent_on_books + additional_books_cost) ∧
  remaining_money_after_DVDs = remaining_money_spent - (fraction * remaining_money_spent + 8) 
  → fraction = 46 / 115 :=
by
  intros
  sorry

end fraction_spent_on_DVDs_l1565_156504


namespace total_area_correct_l1565_156535

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def rect_area : ℝ := length * width
noncomputable def square_side : ℝ := radius * Real.sqrt 2
noncomputable def square_area : ℝ := square_side ^ 2
noncomputable def total_area : ℝ := rect_area + square_area

theorem total_area_correct : total_area = 686 := 
by
  -- Definitions provided above represent the problem's conditions
  -- The value calculated manually is 686
  -- Proof steps skipped for initial statement creation
  sorry

end total_area_correct_l1565_156535


namespace number_of_students_in_range_l1565_156585

-- Define the basic variables and conditions
variable (a b : ℝ) -- Heights of the rectangles in the histogram

-- Define the total number of surveyed students
def total_students : ℝ := 1500

-- Define the width of each histogram group
def group_width : ℝ := 5

-- State the theorem with the conditions and the expected result
theorem number_of_students_in_range (a b : ℝ) :
    5 * (a + b) * total_students = 7500 * (a + b) :=
by
  -- Proof will be added here
  sorry

end number_of_students_in_range_l1565_156585


namespace largest_integer_divides_expression_l1565_156536

theorem largest_integer_divides_expression (x : ℤ) (h : Even x) :
  3 ∣ (10 * x + 1) * (10 * x + 5) * (5 * x + 3) :=
sorry

end largest_integer_divides_expression_l1565_156536


namespace tip_per_person_l1565_156562

-- Define the necessary conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def total_amount_made : ℝ := 37

-- Define the problem statement
theorem tip_per_person : (total_amount_made - hourly_wage) / people_served = 1.25 :=
by
  sorry

end tip_per_person_l1565_156562


namespace sum_sequence_l1565_156538

theorem sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = -2/3)
  (h2 : ∀ n, n ≥ 2 → S n = -1 / (S (n - 1) + 2)) :
  ∀ n, S n = -(n + 1) / (n + 2) := 
by 
  sorry

end sum_sequence_l1565_156538


namespace cubes_with_one_colored_face_l1565_156524

theorem cubes_with_one_colored_face (n : ℕ) (c1 : ℕ) (c2 : ℕ) :
  (n = 64) ∧ (c1 = 4) ∧ (c2 = 4) → ((4 * n) * 2) / n = 32 :=
by 
  sorry

end cubes_with_one_colored_face_l1565_156524


namespace Rachel_brought_25_cookies_l1565_156556

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Total_cookies : ℕ := 60

theorem Rachel_brought_25_cookies : (Total_cookies - (Mona_cookies + Jasmine_cookies) = 25) :=
by
  sorry

end Rachel_brought_25_cookies_l1565_156556


namespace handshakes_min_l1565_156539

-- Define the number of people and the number of handshakes each person performs
def numPeople : ℕ := 35
def handshakesPerPerson : ℕ := 3

-- Define the minimum possible number of unique handshakes
theorem handshakes_min : (numPeople * handshakesPerPerson) / 2 = 105 := by
  sorry

end handshakes_min_l1565_156539


namespace driving_time_l1565_156500

-- Conditions from problem
variable (distance1 : ℕ) (time1 : ℕ) (distance2 : ℕ)
variable (same_speed : distance1 / time1 = distance2 / (5 : ℕ))

-- Statement to prove
theorem driving_time (h1 : distance1 = 120) (h2 : time1 = 3) (h3 : distance2 = 200)
  : distance2 / (40 : ℕ) = (5 : ℕ) := by
  sorry

end driving_time_l1565_156500


namespace Gina_makes_30_per_hour_l1565_156522

variable (rose_cups_per_hour lily_cups_per_hour : ℕ)
variable (rose_cup_order lily_cup_order total_payment : ℕ)
variable (total_hours : ℕ)

def Gina_hourly_rate (rose_cups_per_hour: ℕ) (lily_cups_per_hour: ℕ) (rose_cup_order: ℕ) (lily_cup_order: ℕ) (total_payment: ℕ) : Prop :=
    let rose_time := rose_cup_order / rose_cups_per_hour
    let lily_time := lily_cup_order / lily_cups_per_hour
    let total_time := rose_time + lily_time
    total_payment / total_time = total_hours

theorem Gina_makes_30_per_hour :
    let rose_cups_per_hour := 6
    let lily_cups_per_hour := 7
    let rose_cup_order := 6
    let lily_cup_order := 14
    let total_payment := 90
    Gina_hourly_rate rose_cups_per_hour lily_cups_per_hour rose_cup_order lily_cup_order total_payment 30 :=
by
    sorry

end Gina_makes_30_per_hour_l1565_156522


namespace seat_39_l1565_156552

-- Defining the main structure of the problem
def circle_seating_arrangement (n k : ℕ) : ℕ :=
  if k = 1 then 1
  else sorry -- The pattern-based implementation goes here

-- The theorem to state the problem
theorem seat_39 (n k : ℕ) (h_n : n = 128) (h_k : k = 39) :
  circle_seating_arrangement n k = 51 :=
sorry

end seat_39_l1565_156552


namespace part_I_part_II_l1565_156518

open Real

variable (a b : ℝ)

theorem part_I (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a^2) + (1 / b^2) ≥ 8 := 
sorry

theorem part_II (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end part_I_part_II_l1565_156518


namespace correct_proposition_l1565_156578

def curve_is_ellipse (k : ℝ) : Prop :=
  9 < k ∧ k < 25

def curve_is_hyperbola_on_x_axis (k : ℝ) : Prop :=
  k < 9

theorem correct_proposition (k : ℝ) :
  (curve_is_ellipse k ∨ ¬ curve_is_ellipse k) ∧ 
  (curve_is_hyperbola_on_x_axis k ∨ ¬ curve_is_hyperbola_on_x_axis k) →
  (9 < k ∧ k < 25 → curve_is_ellipse k) ∧ 
  (curve_is_ellipse k ↔ (9 < k ∧ k < 25)) ∧ 
  (curve_is_hyperbola_on_x_axis k ↔ k < 9) → 
  (curve_is_ellipse k ∧ curve_is_hyperbola_on_x_axis k) :=
by
  sorry

end correct_proposition_l1565_156578


namespace find_multiple_l1565_156574

-- Given conditions
variables (P W m : ℕ)
variables (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2)

-- The statement to prove
theorem find_multiple (P W m : ℕ) (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2) : m = 4 :=
by
  sorry

end find_multiple_l1565_156574


namespace lowest_score_l1565_156525

theorem lowest_score (max_mark : ℕ) (n_tests : ℕ) (avg_mark : ℕ) (h_avg : n_tests * avg_mark = 352) (h_max : ∀ k, k < n_tests → k ≤ max_mark) :
  ∃ x, (x ≤ max_mark ∧ (3 * max_mark + x) = 352) ∧ x = 52 :=
by
  sorry

end lowest_score_l1565_156525


namespace students_in_class_l1565_156594

theorem students_in_class (n : ℕ) (T : ℕ) 
  (average_age_students : T = 16 * n)
  (staff_age : ℕ)
  (increased_average_age : (T + staff_age) / (n + 1) = 17)
  (staff_age_val : staff_age = 49) : n = 32 := 
by
  sorry

end students_in_class_l1565_156594


namespace range_of_m_intersection_l1565_156592

noncomputable def f (x m : ℝ) : ℝ := (1/x) - (m/(x^2)) - (x/3)

theorem range_of_m_intersection (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m ∈ (Set.Iic 0 ∪ {2/3}) :=
sorry

end range_of_m_intersection_l1565_156592


namespace central_angle_of_cone_development_diagram_l1565_156551

-- Given conditions: radius of the base of the cone and slant height
def radius_base := 1
def slant_height := 3

-- Target theorem: prove the central angle of the lateral surface development diagram is 120 degrees
theorem central_angle_of_cone_development_diagram : 
  ∃ n : ℝ, (2 * π) = (n * π * slant_height) / 180 ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_cone_development_diagram_l1565_156551


namespace product_of_three_numbers_summing_to_eleven_l1565_156567

def numbers : List ℕ := [2, 3, 4, 6]

theorem product_of_three_numbers_summing_to_eleven : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c = 11 ∧ a * b * c = 36 := 
by
  sorry

end product_of_three_numbers_summing_to_eleven_l1565_156567


namespace triangle_problem_l1565_156512

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def has_same_area (a b : ℕ) (area : ℝ) : Prop :=
  let s := (2 * a + b) / 2
  let areaT := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  areaT = area

def has_same_perimeter (a b : ℕ) (perimeter : ℕ) : Prop :=
  2 * a + b = perimeter

def correct_b (b : ℕ) : Prop :=
  b = 5

theorem triangle_problem
  (a1 a2 b1 b2 : ℕ)
  (h1 : is_isosceles_triangle a1 a1 b1)
  (h2 : is_isosceles_triangle a2 a2 b2)
  (h3 : has_same_area a1 b1 (Real.sqrt 275))
  (h4 : has_same_perimeter a1 b1 22)
  (h5 : has_same_area a2 b2 (Real.sqrt 275))
  (h6 : has_same_perimeter a2 b2 22)
  (h7 : ¬(a1 = a2 ∧ b1 = b2)) : correct_b b2 :=
by
  sorry

end triangle_problem_l1565_156512


namespace spider_distance_l1565_156529

/--
A spider crawls along a number line, starting at -3.
It crawls to -7, then turns around and crawls to 8.
--/
def spiderCrawl (start : ℤ) (point1 : ℤ) (point2 : ℤ): ℤ :=
  let dist1 := abs (point1 - start)
  let dist2 := abs (point2 - point1)
  dist1 + dist2

theorem spider_distance :
  spiderCrawl (-3) (-7) 8 = 19 :=
by
  sorry

end spider_distance_l1565_156529


namespace range_of_a_l1565_156569

theorem range_of_a (a x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 3) (h2 : ∀ x, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) : 1 < a ∧ a < 3 := by
  sorry

end range_of_a_l1565_156569


namespace field_division_l1565_156530

theorem field_division (A B : ℝ) (h1 : A + B = 700) (h2 : B - A = (1 / 5) * ((A + B) / 2)) : A = 315 :=
by
  sorry

end field_division_l1565_156530


namespace surface_area_increase_l1565_156547

theorem surface_area_increase (r h : ℝ) (cs : Bool) : -- cs is a condition switch, True for circular cut, False for rectangular cut
  0 < r ∧ 0 < h →
  let inc_area := if cs then 2 * π * r^2 else 2 * h * r 
  inc_area > 0 :=
by 
  sorry

end surface_area_increase_l1565_156547


namespace odd_product_probability_lt_one_eighth_l1565_156586

theorem odd_product_probability_lt_one_eighth : 
  (∃ p : ℝ, p = (500 / 1000) * (499 / 999) * (498 / 998)) → p < 1 / 8 :=
by
  sorry

end odd_product_probability_lt_one_eighth_l1565_156586


namespace emily_toys_l1565_156568

theorem emily_toys (initial_toys sold_toys: Nat) (h₀ : initial_toys = 7) (h₁ : sold_toys = 3) : initial_toys - sold_toys = 4 := by
  sorry

end emily_toys_l1565_156568


namespace system_solution_l1565_156509

theorem system_solution (x y z a : ℝ) (h1 : x + y + z = 1) (h2 : 1/x + 1/y + 1/z = 1) (h3 : x * y * z = a) :
    (x = 1 ∧ y = Real.sqrt (-a) ∧ z = -Real.sqrt (-a)) ∨
    (x = 1 ∧ y = -Real.sqrt (-a) ∧ z = Real.sqrt (-a)) ∨
    (x = Real.sqrt (-a) ∧ y = -Real.sqrt (-a) ∧ z = 1) ∨
    (x = -Real.sqrt (-a) ∧ y = Real.sqrt (-a) ∧ z = 1) ∨
    (x = Real.sqrt (-a) ∧ y = 1 ∧ z = -Real.sqrt (-a)) ∨
    (x = -Real.sqrt (-a) ∧ y = 1 ∧ z = Real.sqrt (-a)) :=
sorry

end system_solution_l1565_156509


namespace tom_age_ratio_l1565_156549

theorem tom_age_ratio (T N : ℕ) 
  (h1 : T = T)
  (h2 : T - N = 3 * (T - 5 * N)) : T / N = 7 :=
by sorry

end tom_age_ratio_l1565_156549


namespace yura_finishes_on_correct_date_l1565_156543

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end yura_finishes_on_correct_date_l1565_156543


namespace GCF_LCM_computation_l1565_156553

-- Definitions and axioms we need
def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem to prove
theorem GCF_LCM_computation : GCF (LCM 8 14) (LCM 7 12) = 28 :=
by sorry

end GCF_LCM_computation_l1565_156553


namespace bookstore_floor_l1565_156581

theorem bookstore_floor (academy_floor reading_room_floor bookstore_floor : ℤ)
  (h1: academy_floor = 7)
  (h2: reading_room_floor = academy_floor + 4)
  (h3: bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l1565_156581


namespace find_divisor_l1565_156532

theorem find_divisor (dividend remainder quotient : ℕ) (h1 : dividend = 76) (h2 : remainder = 8) (h3 : quotient = 4) : ∃ d : ℕ, dividend = (d * quotient) + remainder ∧ d = 17 :=
by
  sorry

end find_divisor_l1565_156532


namespace eighth_graders_ninth_grader_points_l1565_156517

noncomputable def eighth_grader_points (y : ℚ) (x : ℕ) : Prop :=
  x * y + 8 = ((x + 2) * (x + 1)) / 2

theorem eighth_graders (x : ℕ) (y : ℚ) (hx : eighth_grader_points y x) :
  x = 7 ∨ x = 14 :=
sorry

noncomputable def tenth_grader_points (z y : ℚ) (x : ℕ) : Prop :=
  10 * z = 4.5 * y ∧ x * z = y

theorem ninth_grader_points (y : ℚ) (x : ℕ) (z : ℚ)
  (hx : tenth_grader_points z y x) :
  y = 10 :=
sorry

end eighth_graders_ninth_grader_points_l1565_156517


namespace geometric_sequence_problem_l1565_156576

noncomputable def geometric_sequence_sum_condition 
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 18) ∧ 
  (∀ n, a n = a 1 * q ^ (n - 1)) ∧ 
  (q ≠ 1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h : geometric_sequence_sum_condition a q) : 
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := 
by 
  sorry

end geometric_sequence_problem_l1565_156576


namespace compute_105_squared_l1565_156507

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l1565_156507


namespace ratio_B_A_l1565_156540

theorem ratio_B_A (A B : ℤ) (h : ∀ (x : ℝ), x ≠ -6 → x ≠ 0 → x ≠ 5 → 
  (A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x))) :
  (B : ℚ) / A = 2.2 := by
  sorry

end ratio_B_A_l1565_156540


namespace dewei_less_than_daliah_l1565_156534

theorem dewei_less_than_daliah
  (daliah_amount : ℝ := 17.5)
  (zane_amount : ℝ := 62)
  (zane_multiple_dewei : zane_amount = 4 * (zane_amount / 4)) :
  (daliah_amount - (zane_amount / 4)) = 2 :=
by
  sorry

end dewei_less_than_daliah_l1565_156534


namespace wheel_speed_l1565_156570

theorem wheel_speed (s : ℝ) (t : ℝ) :
  (12 / 5280) * 3600 = s * t →
  (12 / 5280) * 3600 = (s + 4) * (t - (1 / 18000)) →
  s = 8 :=
by
  intro h1 h2
  sorry

end wheel_speed_l1565_156570


namespace condition_on_a_b_l1565_156593

theorem condition_on_a_b (a b : ℝ) (h : a^2 * b^2 + 5 > 2 * a * b - a^2 - 4 * a) : ab ≠ 1 ∨ a ≠ -2 :=
by
  sorry

end condition_on_a_b_l1565_156593


namespace revenue_effect_l1565_156554

noncomputable def price_increase_factor : ℝ := 1.425
noncomputable def sales_decrease_factor : ℝ := 0.627

theorem revenue_effect (P Q R_new : ℝ) (h_price_increase : P ≠ 0) (h_sales_decrease : Q ≠ 0) :
  R_new = (P * price_increase_factor) * (Q * sales_decrease_factor) →
  ((R_new - P * Q) / (P * Q)) * 100 = -10.6825 :=
by
  sorry

end revenue_effect_l1565_156554


namespace vertex_of_parabola_l1565_156566

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, 3 * (x + 4)^2 - 9 = 3 * (x - h)^2 + k) ∧ (h, k) = (-4, -9) :=
by
  sorry

end vertex_of_parabola_l1565_156566


namespace find_rectangle_width_l1565_156582

noncomputable def area_of_square_eq_5times_area_of_rectangle (s l : ℝ) (w : ℝ) :=
  s^2 = 5 * (l * w)

noncomputable def perimeter_of_square_eq_160 (s : ℝ) :=
  4 * s = 160

theorem find_rectangle_width : ∃ w : ℝ, ∀ l : ℝ, 
  area_of_square_eq_5times_area_of_rectangle 40 l w ∧
  perimeter_of_square_eq_160 40 → 
  w = 10 :=
by
  sorry

end find_rectangle_width_l1565_156582


namespace matrix_cubed_l1565_156527

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l1565_156527


namespace financing_term_years_l1565_156519

def monthly_payment : Int := 150
def total_financed_amount : Int := 9000

theorem financing_term_years : 
  (total_financed_amount / monthly_payment) / 12 = 5 := 
by
  sorry

end financing_term_years_l1565_156519


namespace second_person_avg_pages_per_day_l1565_156599

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l1565_156599


namespace spider_total_distance_l1565_156506

theorem spider_total_distance :
  let start := 3
  let mid := -4
  let final := 8
  let dist1 := abs (mid - start)
  let dist2 := abs (final - mid)
  let total_distance := dist1 + dist2
  total_distance = 19 :=
by
  sorry

end spider_total_distance_l1565_156506


namespace product_of_roots_of_quadratic_equation_l1565_156588

theorem product_of_roots_of_quadratic_equation :
  ∀ (x : ℝ), (x^2 + 14 * x + 48 = -4) → (-6) * (-8) = 48 :=
by
  sorry

end product_of_roots_of_quadratic_equation_l1565_156588


namespace symmetry_axis_is_2_range_of_a_l1565_156572

-- Definitions given in the conditions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition 1: Constants a, b, c and a ≠ 0
variables (a b c : ℝ) (a_ne_zero : a ≠ 0)

-- Condition 2: Inequality constraint
axiom inequality_constraint : a^2 + 2 * a * c + c^2 < b^2

-- Condition 3: y-values are the same when x=t+2 and x=-t+2
axiom y_symmetry (t : ℝ) : quadratic_function a b c (t + 2) = quadratic_function a b c (-t + 2)

-- Question 1: Proving the symmetry axis is x=2
theorem symmetry_axis_is_2 : ∀ t : ℝ, (t + 2 + (-t + 2)) / 2 = 2 :=
by sorry

-- Question 2: Proving the range of a if y=2 when x=-2
theorem range_of_a (h : quadratic_function a b c (-2) = 2) (b_eq_neg4a : b = -4 * a) : 2 / 15 < a ∧ a < 2 / 7 :=
by sorry

end symmetry_axis_is_2_range_of_a_l1565_156572


namespace tan_alpha_eq_neg_four_thirds_l1565_156544

theorem tan_alpha_eq_neg_four_thirds
  (α : ℝ) (hα1 : 0 < α ∧ α < π) 
  (hα2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = - 4 / 3 := 
  sorry

end tan_alpha_eq_neg_four_thirds_l1565_156544


namespace perimeter_C_l1565_156541

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l1565_156541


namespace quadratic_trinomial_m_eq_2_l1565_156564

theorem quadratic_trinomial_m_eq_2 (m : ℤ) (P : |m| = 2 ∧ m + 2 ≠ 0) : m = 2 :=
  sorry

end quadratic_trinomial_m_eq_2_l1565_156564


namespace find_w_l1565_156583

theorem find_w (u v w : ℝ) (h1 : 10 * u + 8 * v + 5 * w = 160)
  (h2 : v = u + 3) (h3 : w = 2 * v) : w = 13.5714 := by
  -- The proof would go here, but we leave it empty as per instructions.
  sorry

end find_w_l1565_156583


namespace solve_equation_l1565_156513

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l1565_156513


namespace am_gm_example_l1565_156533

open Real

theorem am_gm_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := 
by 
  sorry

end am_gm_example_l1565_156533


namespace common_ratio_geometric_sequence_l1565_156561

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 1 = 32 ∧ a 6 = -1 → q = -1/2 :=
by
  sorry

end common_ratio_geometric_sequence_l1565_156561


namespace xy_problem_l1565_156521

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l1565_156521


namespace unit_digit_7_power_2023_l1565_156559

theorem unit_digit_7_power_2023 : (7 ^ 2023) % 10 = 3 := by
  sorry

end unit_digit_7_power_2023_l1565_156559


namespace factor_difference_of_squares_l1565_156597

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4 * y) * (9 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l1565_156597


namespace find_number_of_pairs_l1565_156520

variable (n : ℕ)
variable (prob_same_color : ℚ := 0.09090909090909091)
variable (total_shoes : ℕ := 12)
variable (pairs_of_shoes : ℕ)

-- The condition on the probability of selecting two shoes of the same color
def condition_probability : Prop :=
  (1 : ℚ) / ((2 * n - 1) : ℚ) = prob_same_color

-- The condition on the total number of shoes
def condition_total_shoes : Prop :=
  2 * n = total_shoes

-- The goal to prove that the number of pairs of shoes is 6 given the conditions
theorem find_number_of_pairs (h1 : condition_probability n) (h2 : condition_total_shoes n) : n = 6 :=
by
  sorry

end find_number_of_pairs_l1565_156520


namespace abs_neg_three_l1565_156590

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l1565_156590


namespace terminal_side_in_fourth_quadrant_l1565_156523

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ ≥ 0 ∧ θ < Real.pi/2) ∨ (θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :=
sorry

end terminal_side_in_fourth_quadrant_l1565_156523


namespace obtuse_triangle_iff_l1565_156501

theorem obtuse_triangle_iff (x : ℝ) :
    (x > 1 ∧ x < 3) ↔ (x + (x + 1) > (x + 2) ∧
                        (x + 1) + (x + 2) > x ∧
                        (x + 2) + x > (x + 1) ∧
                        (x + 2)^2 > x^2 + (x + 1)^2) :=
by
  sorry

end obtuse_triangle_iff_l1565_156501


namespace function_decreasing_interval_l1565_156557

noncomputable def function_y (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

noncomputable def derivative_y' (x : ℝ) : ℝ := (x + 1) * (x - 1) / x

theorem function_decreasing_interval : ∀ x: ℝ, 0 < x ∧ x < 1 → (derivative_y' x < 0) := by
  sorry

end function_decreasing_interval_l1565_156557


namespace staircase_toothpicks_l1565_156545

theorem staircase_toothpicks (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 9 ∧ r = 3 ∧ n = 3 + 4 
  → (a * r ^ 3 + a * r ^ 2 + a * r + a) + (a * r ^ 2 + a * r + a) + (a * r + a) + a = 351 :=
by
  sorry

end staircase_toothpicks_l1565_156545


namespace probability_more_wins_than_losses_l1565_156550

theorem probability_more_wins_than_losses
  (n_matches : ℕ)
  (win_prob lose_prob tie_prob : ℚ)
  (h_sum_probs : win_prob + lose_prob + tie_prob = 1)
  (h_win_prob : win_prob = 1/3)
  (h_lose_prob : lose_prob = 1/3)
  (h_tie_prob : tie_prob = 1/3)
  (h_n_matches : n_matches = 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m / n = 5483 / 13122 ∧ (m + n) = 18605 :=
by
  sorry

end probability_more_wins_than_losses_l1565_156550


namespace james_birthday_stickers_l1565_156560

def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

def birthday_stickers (s_initial s_final : ℕ) : ℕ := s_final - s_initial

theorem james_birthday_stickers :
  birthday_stickers initial_stickers final_stickers = 22 := by
  sorry

end james_birthday_stickers_l1565_156560


namespace right_triangle_perimeter_l1565_156515

theorem right_triangle_perimeter (a b : ℝ) (c : ℝ) (h1 : a * b = 72) 
  (h2 : c ^ 2 = a ^ 2 + b ^ 2) (h3 : a = 12) :
  a + b + c = 18 + 6 * Real.sqrt 5 := 
by
  sorry

end right_triangle_perimeter_l1565_156515


namespace volume_ratio_of_spheres_l1565_156502

theorem volume_ratio_of_spheres 
  (r1 r2 : ℝ) 
  (h_surface_area : (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 16) : 
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3) = 1 / 64 :=
by 
  sorry

end volume_ratio_of_spheres_l1565_156502


namespace black_beans_count_l1565_156508

theorem black_beans_count (B G O : ℕ) (h₁ : G = B + 2) (h₂ : O = G - 1) (h₃ : B + G + O = 27) : B = 8 := by
  sorry

end black_beans_count_l1565_156508


namespace solution_set_l1565_156598

def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

theorem solution_set : { x : ℝ | f x > 1 } = Set.Ioo (2/3) 2 :=
by
  sorry

end solution_set_l1565_156598


namespace cole_runs_7_miles_l1565_156516

theorem cole_runs_7_miles
  (xavier_miles : ℕ)
  (katie_miles : ℕ)
  (cole_miles : ℕ)
  (h1 : xavier_miles = 3 * katie_miles)
  (h2 : katie_miles = 4 * cole_miles)
  (h3 : xavier_miles = 84)
  (h4 : katie_miles = 28) :
  cole_miles = 7 := 
sorry

end cole_runs_7_miles_l1565_156516


namespace principal_amount_is_26_l1565_156575

-- Define the conditions
def rate : Real := 0.07
def time : Real := 6
def simple_interest : Real := 10.92

-- Define the simple interest formula
def simple_interest_formula (P R T : Real) : Real := P * R * T

-- State the theorem to prove
theorem principal_amount_is_26 : 
  ∃ (P : Real), simple_interest_formula P rate time = simple_interest ∧ P = 26 :=
by
  sorry

end principal_amount_is_26_l1565_156575


namespace quadratic_roots_l1565_156584

theorem quadratic_roots (a b c : ℝ) :
  (∀ (x y : ℝ), ((x, y) = (-2, 12) ∨ (x, y) = (0, -8) ∨ (x, y) = (1, -12) ∨ (x, y) = (3, -8)) → y = a * x^2 + b * x + c) →
  (a * 0^2 + b * 0 + c + 8 = 0) ∧ (a * 3^2 + b * 3 + c + 8 = 0) :=
by sorry

end quadratic_roots_l1565_156584


namespace savanna_more_giraffes_l1565_156514

-- Definitions based on conditions
def lions_safari := 100
def snakes_safari := lions_safari / 2
def giraffes_safari := snakes_safari - 10

def lions_savanna := 2 * lions_safari
def snakes_savanna := 3 * snakes_safari

-- Totals given and to calculate giraffes in Savanna
def total_animals_savanna := 410

-- Prove that Savanna has 20 more giraffes than Safari
theorem savanna_more_giraffes :
  ∃ (giraffes_savanna : ℕ), giraffes_savanna = total_animals_savanna - lions_savanna - snakes_savanna ∧
  giraffes_savanna - giraffes_safari = 20 :=
  by
  sorry

end savanna_more_giraffes_l1565_156514


namespace unique_solution_for_system_l1565_156563

theorem unique_solution_for_system (a : ℝ) :
  (∃! (x y : ℝ), x^2 + 4 * y^2 = 1 ∧ x + 2 * y = a) ↔ a = -1.41 :=
by
  sorry

end unique_solution_for_system_l1565_156563


namespace min_value_reciprocal_l1565_156505

theorem min_value_reciprocal (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_reciprocal_l1565_156505


namespace x_intercept_is_3_l1565_156596

-- Define the given points
def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (6, 6)

-- Prove the x-intercept is 3
theorem x_intercept_is_3 (x : ℝ) :
  (∃ m b : ℝ, (∀ x1 y1 x2 y2 : ℝ, (y1 = m * x1 + b) ∧ (x1, y1) = point1 ∧ (x2, y2) = point2) ∧ y = 0 ∧ x = -b / m) → x = 3 :=
sorry

end x_intercept_is_3_l1565_156596


namespace smallest_m_satisfying_conditions_l1565_156510

theorem smallest_m_satisfying_conditions :
  ∃ m : ℕ, m = 4 ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ m ∧ (m^2 + m) % k ≠ 0) ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ m) → (k ≠ 0 → (m^2 + m) % k = 0)) :=
sorry

end smallest_m_satisfying_conditions_l1565_156510


namespace rational_solutions_k_l1565_156546

theorem rational_solutions_k (k : ℕ) (hpos : k > 0) : (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 :=
by
  sorry

end rational_solutions_k_l1565_156546


namespace initial_percentage_increase_l1565_156591

-- Given conditions
def S_original : ℝ := 4000.0000000000005
def S_final : ℝ := 4180
def reduction : ℝ := 5

-- Predicate to prove the initial percentage increase is 10%
theorem initial_percentage_increase (x : ℝ) 
  (hx : (95/100) * (S_original * (1 + x / 100)) = S_final) : 
  x = 10 :=
sorry

end initial_percentage_increase_l1565_156591


namespace least_subtraction_l1565_156579

theorem least_subtraction (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 45678) (h2 : d = 47) (h3 : n % d = r) : r = 35 :=
by {
  sorry
}

end least_subtraction_l1565_156579


namespace factor_expression_l1565_156580

theorem factor_expression (x : ℝ) : 
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) :=
by 
  sorry

end factor_expression_l1565_156580


namespace investment_period_l1565_156511

theorem investment_period (P : ℝ) (r1 r2 : ℝ) (diff : ℝ) (t : ℝ) :
  P = 900 ∧ r1 = 0.04 ∧ r2 = 0.045 ∧ (P * r2 * t) - (P * r1 * t) = 31.50 → t = 7 :=
by
  sorry

end investment_period_l1565_156511


namespace harold_wrapping_paper_cost_l1565_156526

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end harold_wrapping_paper_cost_l1565_156526


namespace spider_total_distance_l1565_156528

theorem spider_total_distance : 
  ∀ (pos1 pos2 pos3 : ℝ), pos1 = 3 → pos2 = -1 → pos3 = 8.5 → 
  |pos2 - pos1| + |pos3 - pos2| = 13.5 := 
by 
  intros pos1 pos2 pos3 hpos1 hpos2 hpos3 
  sorry

end spider_total_distance_l1565_156528


namespace intersection_A_complement_B_l1565_156548

-- Definition of real numbers
def R := ℝ

-- Definitions of sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x^2 - x - 2 > 0}

-- Definition of the complement of B in R
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- The final statement we need to prove
theorem intersection_A_complement_B :
  A ∩ B_complement = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_complement_B_l1565_156548


namespace find_ellipse_and_hyperbola_equations_l1565_156577

-- Define the conditions
def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = (Real.sqrt (a ^ 2 - b ^ 2)) / a

def focal_distance (f : ℝ) (a b : ℝ) : Prop :=
  f = 2 * Real.sqrt (a ^ 2 + b ^ 2)

-- Define the problem to prove the equations of the ellipse and hyperbola
theorem find_ellipse_and_hyperbola_equations (a b : ℝ) (e : ℝ) (f : ℝ)
  (h1 : eccentricity e a b) (h2 : focal_distance f a b) 
  (h3 : e = 4 / 5) (h4 : f = 2 * Real.sqrt 34) 
  (h5 : a > b) (h6 : 0 < b) :
  (a^2 = 25 ∧ b^2 = 9) → 
  (∀ x y, (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 / 25 - y^2 / 9 = 1)) :=
sorry

end find_ellipse_and_hyperbola_equations_l1565_156577


namespace find_m_n_l1565_156531

theorem find_m_n (m n : ℤ) (h : |m - 2| + (n^2 - 8 * n + 16) = 0) : m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l1565_156531


namespace rate_in_still_water_l1565_156537

-- Definitions of given conditions
def downstream_speed : ℝ := 26
def upstream_speed : ℝ := 12

-- The statement we need to prove
theorem rate_in_still_water : (downstream_speed + upstream_speed) / 2 = 19 := by
  sorry

end rate_in_still_water_l1565_156537


namespace triangle_equilateral_l1565_156558

noncomputable def is_equilateral {R p : ℝ} (A B C : ℝ) : Prop :=
  R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p  →
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a

theorem triangle_equilateral
  {A B C : ℝ}
  {R p : ℝ}
  (h : R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p) :
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a :=
sorry

end triangle_equilateral_l1565_156558


namespace configuration_count_l1565_156565

theorem configuration_count :
  (∃ (w h s : ℕ), 2 * (w + h + 2 * s) = 120 ∧ w < h ∧ s % 2 = 0) →
  ∃ n, n = 196 := 
sorry

end configuration_count_l1565_156565


namespace find_number_l1565_156503

theorem find_number (x : ℕ) (h : 23 + x = 34) : x = 11 :=
by
  sorry

end find_number_l1565_156503


namespace compound_interest_rate_l1565_156555

theorem compound_interest_rate (P A : ℝ) (t n : ℝ)
  (hP : P = 5000) 
  (hA : A = 7850)
  (ht : t = 8)
  (hn : n = 1) : 
  ∃ r : ℝ, 0.057373 ≤ (r * 100) ∧ (r * 100) ≤ 5.7373 :=
by
  sorry

end compound_interest_rate_l1565_156555


namespace divisibility_of_n_l1565_156542

def n : ℕ := (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1)

theorem divisibility_of_n : 
    (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) := 
by 
  sorry

end divisibility_of_n_l1565_156542


namespace find_other_vertices_l1565_156573

theorem find_other_vertices
  (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (S : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (7, 3))
  (hS : S = (5, -5 / 3))
  (hM : M = (3, -1))
  (h_centroid : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) 
  (h_orthocenter : ∀ u v : ℝ × ℝ, u ≠ v → u - v = (4, 4) → (u - v) • (C - B) = 0) :
  B = (1, -1) ∧ C = (7, -7) :=
sorry

end find_other_vertices_l1565_156573


namespace maria_total_flowers_l1565_156571

-- Define the initial conditions
def dozens := 3
def flowers_per_dozen := 12
def free_flowers_per_dozen := 2

-- Define the total number of flowers
def total_flowers := dozens * flowers_per_dozen + dozens * free_flowers_per_dozen

-- Assert the proof statement
theorem maria_total_flowers : total_flowers = 42 := sorry

end maria_total_flowers_l1565_156571
