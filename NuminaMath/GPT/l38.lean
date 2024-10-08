import Mathlib

namespace percent_increase_calculation_l38_38290

variable (x y : ℝ) -- Declare x and y as real numbers representing the original salary and increment

-- The statement that the percent increase z follows from the given conditions
theorem percent_increase_calculation (h : y + x = x + y) : (y / x) * 100 = ((y / x) * 100) := by
  sorry

end percent_increase_calculation_l38_38290


namespace geometric_sum_eqn_l38_38822

theorem geometric_sum_eqn 
  (a1 q : ℝ) 
  (hne1 : q ≠ 1) 
  (hS2 : a1 * (1 - q^2) / (1 - q) = 1) 
  (hS4 : a1 * (1 - q^4) / (1 - q) = 3) :
  a1 * (1 - q^8) / (1 - q) = 15 :=
by
  sorry

end geometric_sum_eqn_l38_38822


namespace only_valid_set_is_b_l38_38385

def can_form_triangle (a b c : Nat) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem only_valid_set_is_b :
  can_form_triangle 2 3 4 ∧ 
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 4 9 ∧
  ¬ can_form_triangle 2 2 4 := by
  sorry

end only_valid_set_is_b_l38_38385


namespace sum_of_products_of_roots_l38_38737

theorem sum_of_products_of_roots (p q r : ℂ) (h : 4 * (p^3) - 2 * (p^2) + 13 * p - 9 = 0 ∧ 4 * (q^3) - 2 * (q^2) + 13 * q - 9 = 0 ∧ 4 * (r^3) - 2 * (r^2) + 13 * r - 9 = 0) :
  p*q + p*r + q*r = 13 / 4 :=
  sorry

end sum_of_products_of_roots_l38_38737


namespace correct_equation_l38_38459

def initial_investment : ℝ := 2500
def expected_investment : ℝ := 6600
def growth_rate (x : ℝ) : ℝ := x

theorem correct_equation (x : ℝ) : 
  initial_investment * (1 + growth_rate x) + initial_investment * (1 + growth_rate x)^2 = expected_investment :=
by
  sorry

end correct_equation_l38_38459


namespace positive_number_property_l38_38170

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_eq : (x^2) / 100 = 9) : x = 30 :=
sorry

end positive_number_property_l38_38170


namespace range_of_x_l38_38626

theorem range_of_x (x : ℝ) (h1 : (x + 2) * (x - 3) ≤ 0) (h2 : |x + 1| ≥ 2) : 
  1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l38_38626


namespace smallest_k_condition_exists_l38_38393

theorem smallest_k_condition_exists (k : ℕ) :
    k > 1 ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 3 = 1) → k = 313 :=
by
  sorry

end smallest_k_condition_exists_l38_38393


namespace digits_difference_l38_38574

-- Definitions based on conditions
variables (X Y : ℕ)

-- Condition: The difference between the original number and the interchanged number is 27
def difference_condition : Prop :=
  (10 * X + Y) - (10 * Y + X) = 27

-- Problem to prove: The difference between the two digits is 3
theorem digits_difference (h : difference_condition X Y) : X - Y = 3 :=
by sorry

end digits_difference_l38_38574


namespace problem1_problem2_l38_38122

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l38_38122


namespace vertex_on_x_axis_l38_38391

theorem vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 6 * x + d = 0) ↔ d = 9 :=
by
  sorry

end vertex_on_x_axis_l38_38391


namespace range_of_m_min_of_squares_l38_38100

-- 1. Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 4)

-- 2. State the condition that f(x) ≤ -m^2 + 6m holds for all x
def condition (m : ℝ) : Prop := ∀ x : ℝ, f x ≤ -m^2 + 6 * m

-- 3. State the range of m to be proven
theorem range_of_m : ∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5 := 
sorry

-- 4. Auxiliary condition for part 2
def m_0 : ℝ := 5

-- 5. State the condition 3a + 4b + 5c = m_0
def sum_condition (a b c : ℝ) : Prop := 3 * a + 4 * b + 5 * c = m_0

-- 6. State the minimum value problem
theorem min_of_squares (a b c : ℝ) : sum_condition a b c → a^2 + b^2 + c^2 ≥ 1 / 2 := 
sorry

end range_of_m_min_of_squares_l38_38100


namespace mean_properties_l38_38514

theorem mean_properties (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (arith_mean : (x + y + z) / 3 = 10)
  (geom_mean : (x * y * z) ^ (1 / 3) = 6)
  (harm_mean : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := 
sorry

end mean_properties_l38_38514


namespace monkey_reaches_top_l38_38778

def monkey_climb_time (tree_height : ℕ) (climb_per_hour : ℕ) (slip_per_hour : ℕ) 
  (rest_hours : ℕ) (cycle_hours : ℕ) : ℕ :=
  if (tree_height % (climb_per_hour - slip_per_hour) > climb_per_hour) 
    then (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours
    else (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours - 1

theorem monkey_reaches_top :
  monkey_climb_time 253 7 4 1 4 = 109 := 
sorry

end monkey_reaches_top_l38_38778


namespace printer_x_time_l38_38093

-- Define the basic parameters given in the problem
def job_time_printer_y := 12
def job_time_printer_z := 8
def ratio := 10 / 3

-- Work rates of the printers
def work_rate_y := 1 / job_time_printer_y
def work_rate_z := 1 / job_time_printer_z

-- Combined work rate and total time for printers Y and Z
def combined_work_rate_y_z := work_rate_y + work_rate_z
def time_printers_y_z := 1 / combined_work_rate_y_z

-- Given ratio relation
def time_printer_x := ratio * time_printers_y_z

-- Mathematical statement to prove: time it takes for printer X to do the job alone
theorem printer_x_time : time_printer_x = 16 := by
  sorry

end printer_x_time_l38_38093


namespace point_in_quadrants_l38_38808

theorem point_in_quadrants (x y : ℝ) (h1 : 4 * x + 7 * y = 28) (h2 : |x| = |y|) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  sorry

end point_in_quadrants_l38_38808


namespace find_vector_at_t_zero_l38_38688

variable (a d : ℝ × ℝ × ℝ)
variable (t : ℝ)

-- Given conditions
def condition1 := a - 2 * d = (2, 4, 10)
def condition2 := a + d = (-1, -3, -5)

-- The proof problem
theorem find_vector_at_t_zero 
  (h1 : condition1 a d)
  (h2 : condition2 a d) :
  a = (0, -2/3, 0) :=
sorry

end find_vector_at_t_zero_l38_38688


namespace right_triangle_area_l38_38867

theorem right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) 
  (h_angle_sum : a = 45) (h_other_angle : b = 45) (h_right_angle : c = 90)
  (h_altitude : ∃ height : ℝ, height = 4) :
  ∃ area : ℝ, area = 8 := 
by
  sorry

end right_triangle_area_l38_38867


namespace slope_angle_vertical_line_l38_38772

theorem slope_angle_vertical_line : 
  ∀ α : ℝ, (∀ x y : ℝ, x = 1 → y = α) → α = Real.pi / 2 := 
by 
  sorry

end slope_angle_vertical_line_l38_38772


namespace books_per_shelf_l38_38430

theorem books_per_shelf (total_distance : ℕ) (total_shelves : ℕ) (one_way_distance : ℕ) 
  (h1 : total_distance = 3200) (h2 : total_shelves = 4) (h3 : one_way_distance = total_distance / 2) 
  (h4 : one_way_distance = 1600) :
  ∀ books_per_shelf : ℕ, books_per_shelf = one_way_distance / total_shelves := 
by
  sorry

end books_per_shelf_l38_38430


namespace determine_range_of_m_l38_38540

noncomputable def range_m (m : ℝ) (x : ℝ) : Prop :=
  ∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
       (∃ x, -x^2 + 7 * x + 8 ≥ 0)

theorem determine_range_of_m (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 1) ↔
  (∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
         (∃ x, -x^2 + 7 * x + 8 ≥ 0)) :=
by
  sorry

end determine_range_of_m_l38_38540


namespace norm_of_w_l38_38174

variable (u v : EuclideanSpace ℝ (Fin 2)) 
variable (hu : ‖u‖ = 3) (hv : ‖v‖ = 5) 
variable (h_orthogonal : inner u v = 0)

theorem norm_of_w :
  ‖4 • u - 2 • v‖ = 2 * Real.sqrt 61 := by
  sorry

end norm_of_w_l38_38174


namespace base_7_to_10_of_23456_l38_38080

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l38_38080


namespace find_down_payment_l38_38471

noncomputable def purchasePrice : ℝ := 118
noncomputable def monthlyPayment : ℝ := 10
noncomputable def numberOfMonths : ℝ := 12
noncomputable def interestRate : ℝ := 0.15254237288135593
noncomputable def totalPayments : ℝ := numberOfMonths * monthlyPayment -- total amount paid through installments
noncomputable def interestPaid : ℝ := purchasePrice * interestRate -- total interest paid
noncomputable def totalPaid : ℝ := purchasePrice + interestPaid -- total amount paid including interest

theorem find_down_payment : ∃ D : ℝ, D + totalPayments = totalPaid ∧ D = 16 :=
by sorry

end find_down_payment_l38_38471


namespace cookout_ratio_l38_38978

theorem cookout_ratio (K_2004 K_2005 : ℕ) (h1 : K_2004 = 60) (h2 : (2 / 3) * K_2005 = 20) :
  K_2005 / K_2004 = 1 / 2 :=
by sorry

end cookout_ratio_l38_38978


namespace transformed_stats_l38_38550

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt ((l.map (λ x => (x - mean l)^2)).sum / l.length)

theorem transformed_stats (l : List ℝ) 
  (hmean : mean l = 10)
  (hstddev : std_dev l = 2) :
  mean (l.map (λ x => 2 * x - 1)) = 19 ∧ std_dev (l.map (λ x => 2 * x - 1)) = 4 := by
  sorry

end transformed_stats_l38_38550


namespace vector_combination_l38_38344

-- Define the vectors and the conditions
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * u.1, k * u.2)

-- The main theorem to be proved
theorem vector_combination (m : ℝ) (h_parallel : parallel vec_a (vec_b m)) : 3 * vec_a + 2 * vec_b m = (7, -14) := by
  sorry

end vector_combination_l38_38344


namespace problem_remainder_l38_38828

theorem problem_remainder :
  ((12095 + 12097 + 12099 + 12101 + 12103 + 12105 + 12107) % 10) = 7 := by
  sorry

end problem_remainder_l38_38828


namespace final_volume_of_water_in_tank_l38_38878

theorem final_volume_of_water_in_tank (capacity : ℕ) (initial_fraction full_volume : ℕ)
  (percent_empty percent_fill final_volume : ℕ) :
  capacity = 8000 →
  initial_fraction = 3 / 4 →
  percent_empty = 40 →
  percent_fill = 30 →
  full_volume = capacity * initial_fraction →
  final_volume = full_volume - (full_volume * percent_empty / 100) + ((full_volume - (full_volume * percent_empty / 100)) * percent_fill / 100) →
  final_volume = 4680 :=
by
  sorry

end final_volume_of_water_in_tank_l38_38878


namespace power_of_three_l38_38946

theorem power_of_three (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_mult : (3^a) * (3^b) = 81) : (3^a)^b = 81 :=
sorry

end power_of_three_l38_38946


namespace pool_filling_time_l38_38992

noncomputable def fill_pool_time (hose_rate : ℕ) (cost_per_10_gallons : ℚ) (total_cost : ℚ) : ℚ :=
  let cost_per_gallon := cost_per_10_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_rate

theorem pool_filling_time :
  fill_pool_time 100 (1 / 100) 5 = 50 := 
by
  sorry

end pool_filling_time_l38_38992


namespace new_person_weight_l38_38891

theorem new_person_weight (W : ℝ) (N : ℝ)
  (h1 : ∀ avg_increase : ℝ, avg_increase = 2.5 → N = 55) 
  (h2 : ∀ original_weight : ℝ, original_weight = 35) 
  : N = 55 := 
by 
  sorry

end new_person_weight_l38_38891


namespace total_shaded_area_is_71_l38_38662

-- Define the dimensions of the first rectangle
def rect1_length : ℝ := 4
def rect1_width : ℝ := 12

-- Define the dimensions of the second rectangle
def rect2_length : ℝ := 5
def rect2_width : ℝ := 7

-- Define the dimensions of the overlap area
def overlap_length : ℝ := 3
def overlap_width : ℝ := 4

-- Define the area calculation
def area (length width : ℝ) : ℝ := length * width

-- Calculate the areas of the rectangles and the overlap
def rect1_area : ℝ := area rect1_length rect1_width
def rect2_area : ℝ := area rect2_length rect2_width
def overlap_area : ℝ := area overlap_length overlap_width

-- Total shaded area calculation
def total_shaded_area : ℝ := rect1_area + rect2_area - overlap_area

-- Proof statement to show that the total shaded area is 71 square units
theorem total_shaded_area_is_71 : total_shaded_area = 71 := by
  sorry

end total_shaded_area_is_71_l38_38662


namespace totalNameLengths_l38_38020

-- Definitions of the lengths of names
def JonathanNameLength := 8 + 10
def YoungerSisterNameLength := 5 + 10
def OlderBrotherNameLength := 6 + 10
def YoungestSisterNameLength := 4 + 15

-- Statement to prove
theorem totalNameLengths :
  JonathanNameLength + YoungerSisterNameLength + OlderBrotherNameLength + YoungestSisterNameLength = 68 :=
by
  sorry -- no proof required

end totalNameLengths_l38_38020


namespace train_passes_man_in_correct_time_l38_38136

-- Definitions for the given conditions
def platform_length : ℝ := 270
def train_length : ℝ := 180
def crossing_time : ℝ := 20

-- Theorem to prove the time taken to pass the man is 8 seconds
theorem train_passes_man_in_correct_time
  (p: ℝ) (l: ℝ) (t_cross: ℝ)
  (h1: p = platform_length)
  (h2: l = train_length)
  (h3: t_cross = crossing_time) :
  l / ((l + p) / t_cross) = 8 := by
  -- Proof goes here
  sorry

end train_passes_man_in_correct_time_l38_38136


namespace min_value_l38_38556

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ xy : ℝ, (xy = 9 ∧ (forall (u v : ℝ), (u > 0) → (v > 0) → 2 * u + v = 1 → (2 / u) + (1 / v) ≥ xy)) :=
by
  use 9
  sorry

end min_value_l38_38556


namespace Jake_peach_count_l38_38786

theorem Jake_peach_count (Steven_peaches : ℕ) (Jake_peach_difference : ℕ) (h1 : Steven_peaches = 19) (h2 : Jake_peach_difference = 12) : 
  Steven_peaches - Jake_peach_difference = 7 :=
by
  sorry

end Jake_peach_count_l38_38786


namespace minimum_distance_on_line_l38_38235

-- Define the line as a predicate
def on_line (P : ℝ × ℝ) : Prop := P.1 - P.2 = 1

-- Define the expression to be minimized
def distance_squared (P : ℝ × ℝ) : ℝ := (P.1 - 2)^2 + (P.2 - 2)^2

theorem minimum_distance_on_line :
  ∃ P : ℝ × ℝ, on_line P ∧ distance_squared P = 1 / 2 :=
sorry

end minimum_distance_on_line_l38_38235


namespace percentage_return_l38_38489

theorem percentage_return (income investment : ℝ) (h_income : income = 680) (h_investment : investment = 8160) :
  (income / investment) * 100 = 8.33 :=
by
  rw [h_income, h_investment]
  -- The rest of the proof is omitted.
  sorry

end percentage_return_l38_38489


namespace savings_after_expense_increase_l38_38368

-- Define the conditions
def monthly_salary : ℝ := 6500
def initial_savings_percentage : ℝ := 0.20
def increase_expenses_percentage : ℝ := 0.20

-- Define the statement we want to prove
theorem savings_after_expense_increase :
  (monthly_salary - (monthly_salary - (initial_savings_percentage * monthly_salary) + (increase_expenses_percentage * (monthly_salary - (initial_savings_percentage * monthly_salary))))) = 260 :=
sorry

end savings_after_expense_increase_l38_38368


namespace black_square_area_l38_38036

-- Define the edge length of the cube
def edge_length := 12

-- Define the total amount of yellow paint available
def yellow_paint_area := 432

-- Define the total surface area of the cube
def total_surface_area := 6 * (edge_length * edge_length)

-- Define the area covered by yellow paint per face
def yellow_per_face := yellow_paint_area / 6

-- Define the area of one face of the cube
def face_area := edge_length * edge_length

-- State the theorem: the area of the black square on each face
theorem black_square_area : (face_area - yellow_per_face) = 72 := by
  sorry

end black_square_area_l38_38036


namespace center_of_circle_l38_38222

theorem center_of_circle :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → (x, y) = (1, 1) :=
by
  sorry

end center_of_circle_l38_38222


namespace box_weight_in_kg_l38_38693

def weight_of_one_bar : ℕ := 125 -- Weight of one chocolate bar in grams
def number_of_bars : ℕ := 16 -- Number of chocolate bars in the box
def grams_to_kg (g : ℕ) : ℕ := g / 1000 -- Function to convert grams to kilograms

theorem box_weight_in_kg : grams_to_kg (weight_of_one_bar * number_of_bars) = 2 :=
by
  sorry -- Proof is omitted

end box_weight_in_kg_l38_38693


namespace rectangle_dimensions_l38_38359

theorem rectangle_dimensions (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * (l + w) = 3 * (l * w)) : 
  w = 1 ∧ l = 2 := by
  sorry

end rectangle_dimensions_l38_38359


namespace man_age_difference_l38_38040

theorem man_age_difference (S M : ℕ) (h1 : S = 24) (h2 : M + 2 = 2 * (S + 2)) : M - S = 26 := by
  sorry

end man_age_difference_l38_38040


namespace sin_minus_cos_eq_one_l38_38663

theorem sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) : x = Real.pi / 2 :=
by sorry

end sin_minus_cos_eq_one_l38_38663


namespace minimum_value_l38_38934

-- Define the expression E(a, b, c)
def E (a b c : ℝ) : ℝ := a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2

-- State the minimum value theorem
theorem minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  E a b c = 18 :=
sorry

end minimum_value_l38_38934


namespace remaining_cookies_l38_38108

theorem remaining_cookies : 
  let naomi_cookies := 53
  let oliver_cookies := 67
  let penelope_cookies := 29
  let total_cookies := naomi_cookies + oliver_cookies + penelope_cookies
  let package_size := 15
  total_cookies % package_size = 14 :=
by
  sorry

end remaining_cookies_l38_38108


namespace probability_sum_even_is_five_over_eleven_l38_38536

noncomputable def probability_even_sum : ℚ :=
  let totalBalls := 12
  let totalWays := totalBalls * (totalBalls - 1)
  let evenBalls := 6
  let oddBalls := 6
  let evenWays := evenBalls * (evenBalls - 1)
  let oddWays := oddBalls * (oddBalls - 1)
  let totalEvenWays := evenWays + oddWays
  totalEvenWays / totalWays

theorem probability_sum_even_is_five_over_eleven : probability_even_sum = 5 / 11 := sorry

end probability_sum_even_is_five_over_eleven_l38_38536


namespace roberto_valid_outfits_l38_38487

-- Definitions based on the conditions
def total_trousers : ℕ := 6
def total_shirts : ℕ := 8
def total_jackets : ℕ := 4
def restricted_jacket : ℕ := 1
def restricted_shirts : ℕ := 2

-- Theorem statement
theorem roberto_valid_outfits : 
  total_trousers * total_shirts * total_jackets - total_trousers * restricted_shirts * restricted_jacket = 180 := 
by
  sorry

end roberto_valid_outfits_l38_38487


namespace find_x_l38_38124

theorem find_x (x : ℕ) (h : 2^x - 2^(x - 2) = 3 * 2^10) : x = 12 :=
by 
  sorry

end find_x_l38_38124


namespace total_pencils_l38_38284

theorem total_pencils (reeta_pencils anika_pencils kamal_pencils : ℕ) :
  reeta_pencils = 30 →
  anika_pencils = 2 * reeta_pencils + 4 →
  kamal_pencils = 3 * reeta_pencils - 2 →
  reeta_pencils + anika_pencils + kamal_pencils = 182 :=
by
  intros h_reeta h_anika h_kamal
  sorry

end total_pencils_l38_38284


namespace positive_inequality_l38_38198

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * (a^2 * b^2) / 4 + 9 * a * b > 18 := 
  sorry

end positive_inequality_l38_38198


namespace total_driving_time_is_40_l38_38813

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l38_38813


namespace favorite_movies_hours_l38_38568

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l38_38568


namespace children_gift_distribution_l38_38301

theorem children_gift_distribution (N : ℕ) (hN : N > 1) :
  (∀ n : ℕ, n < N → (∃ k : ℕ, k < N ∧ k ≠ n)) →
  (∃ m : ℕ, (N - 1) = 2 * m) :=
by
  sorry

end children_gift_distribution_l38_38301


namespace sammy_pickles_l38_38099

theorem sammy_pickles 
  (T S R : ℕ) 
  (h1 : T = 2 * S) 
  (h2 : R = 8 * T / 10) 
  (h3 : R = 24) : 
  S = 15 :=
by
  sorry

end sammy_pickles_l38_38099


namespace weight_of_b_l38_38535

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : B = 51 := 
by
  sorry

end weight_of_b_l38_38535


namespace sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l38_38440

-- Definitions
def a (t : ℤ) := 4 * t
def b (t : ℤ) := 3 - 2 * t - t^2
def c (t : ℤ) := 3 + 2 * t - t^2

-- Theorem for sum of squares
theorem sum_of_squares_twice_square (t : ℤ) : 
  a t ^ 2 + b t ^ 2 + c t ^ 2 = 2 * ((3 + t^2) ^ 2) :=
by 
  sorry

-- Theorem for sum of fourth powers
theorem sum_of_fourth_powers_twice_fourth_power (t : ℤ) : 
  a t ^ 4 + b t ^ 4 + c t ^ 4 = 2 * ((3 + t^2) ^ 4) :=
by 
  sorry

end sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l38_38440


namespace ratio_of_white_socks_l38_38636

theorem ratio_of_white_socks 
  (total_socks : ℕ) (blue_socks : ℕ)
  (h_total_socks : total_socks = 180)
  (h_blue_socks : blue_socks = 60) :
  (total_socks - blue_socks) * 3 = total_socks * 2 :=
by
  sorry

end ratio_of_white_socks_l38_38636


namespace decompose_one_into_five_unit_fractions_l38_38172

theorem decompose_one_into_five_unit_fractions :
  1 = (1/2) + (1/3) + (1/7) + (1/43) + (1/1806) :=
by
  sorry

end decompose_one_into_five_unit_fractions_l38_38172


namespace zoo_guides_children_total_l38_38596

theorem zoo_guides_children_total :
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  total_children = 1674 :=
by
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  sorry

end zoo_guides_children_total_l38_38596


namespace lindsey_integer_l38_38625

theorem lindsey_integer (n : ℕ) (a b c : ℤ) (h1 : n < 50)
                        (h2 : n = 6 * a - 1)
                        (h3 : n = 8 * b - 5)
                        (h4 : n = 3 * c + 2) :
  n = 41 := 
  by sorry

end lindsey_integer_l38_38625


namespace jill_sod_area_needed_l38_38639

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l38_38639


namespace reflection_y_axis_matrix_correct_l38_38767

def reflect_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]]

theorem reflection_y_axis_matrix_correct :
  reflect_y_axis_matrix = ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]] :=
by
  sorry

end reflection_y_axis_matrix_correct_l38_38767


namespace polygon_sides_with_diagonals_44_l38_38282

theorem polygon_sides_with_diagonals_44 (n : ℕ) (hD : 44 = n * (n - 3) / 2) : n = 11 :=
by
  sorry

end polygon_sides_with_diagonals_44_l38_38282


namespace john_new_weekly_earnings_l38_38205

theorem john_new_weekly_earnings
  (original_earnings : ℕ)
  (percentage_increase : ℕ)
  (raise_amount : ℕ)
  (new_weekly_earnings : ℕ)
  (original_earnings_eq : original_earnings = 50)
  (percentage_increase_eq : percentage_increase = 40)
  (raise_amount_eq : raise_amount = original_earnings * percentage_increase / 100)
  (new_weekly_earnings_eq : new_weekly_earnings = original_earnings + raise_amount) :
  new_weekly_earnings = 70 := by
  sorry

end john_new_weekly_earnings_l38_38205


namespace sum_of_coordinates_of_intersection_l38_38369

def h : ℝ → ℝ := -- Define h(x). This would be specific to the function provided; we abstract it here for the proof.
sorry

theorem sum_of_coordinates_of_intersection (a b : ℝ) (h_eq: h a = h (a - 5)) : a + b = 6 :=
by
  -- We need a [step from the problem conditions], hence introducing the given conditions
  have : b = h a := sorry
  have : b = h (a - 5) := sorry
  exact sorry

end sum_of_coordinates_of_intersection_l38_38369


namespace tangent_line_parabola_l38_38328

theorem tangent_line_parabola (k : ℝ) 
  (h : ∀ (x y : ℝ), 4 * x + 6 * y + k = 0 → y^2 = 32 * x) : k = 72 := 
sorry

end tangent_line_parabola_l38_38328


namespace exponent_equality_l38_38612

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l38_38612


namespace sum_of_three_largest_l38_38853

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l38_38853


namespace difference_between_wins_and_losses_l38_38076

noncomputable def number_of_wins (n m : ℕ) : Prop :=
  0 ≤ n ∧ 0 ≤ m ∧ n + m ≤ 42 ∧ n + (42 - n - m) / 2 = 30 / 1

theorem difference_between_wins_and_losses (n m : ℕ) (h : number_of_wins n m) : n - m = 18 :=
sorry

end difference_between_wins_and_losses_l38_38076


namespace max_ab_condition_max_ab_value_l38_38498

theorem max_ab_condition (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a + b = 1) (h2 : a = b) : ab = 1 / 4 :=
sorry

end max_ab_condition_max_ab_value_l38_38498


namespace molecular_weight_of_compound_l38_38224

def hydrogen_atomic_weight : ℝ := 1.008
def chromium_atomic_weight : ℝ := 51.996
def oxygen_atomic_weight : ℝ := 15.999

def compound_molecular_weight (h_atoms : ℕ) (cr_atoms : ℕ) (o_atoms : ℕ) : ℝ :=
  h_atoms * hydrogen_atomic_weight + cr_atoms * chromium_atomic_weight + o_atoms * oxygen_atomic_weight

theorem molecular_weight_of_compound :
  compound_molecular_weight 2 1 4 = 118.008 :=
by
  sorry

end molecular_weight_of_compound_l38_38224


namespace extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l38_38724

-- Given function
def f (x : ℝ) : ℝ := x^3 - 12 * x + 12

-- Definition of derivative
def f' (x : ℝ) : ℝ := (3 : ℝ) * x^2 - (12 : ℝ)

-- Part 1: Extreme values
theorem extreme_values : 
  (f (-2) = 28) ∧ (f 2 = -4) :=
by
  sorry

-- Part 2: Maximum and minimum values on the interval [-3, 4]
theorem max_min_on_interval :
  (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧ (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≥ -4) :=
by
  sorry

-- Part 3: Coordinates of midpoint A and B with parallel tangents
theorem coordinates_midpoint_parallel_tangents :
  (f' x1 = f' x2 ∧ x1 + x2 = 0) → ((x1 + x2) / 2 = 0 ∧ (f x1 + f x2) / 2 = 12) :=
by
  sorry

end extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l38_38724


namespace arithmetic_sequence_1001th_term_l38_38858

theorem arithmetic_sequence_1001th_term (p q : ℚ)
    (h1 : p + 3 * q = 12)
    (h2 : 12 + 3 * q = 3 * p - q) :
    (p + (1001 - 1) * (3 * q) = 5545) :=
by
  sorry

end arithmetic_sequence_1001th_term_l38_38858


namespace max_g_value_l38_38543

def g (n : ℕ) : ℕ :=
if h : n < 10 then 2 * n + 3 else g (n - 7)

theorem max_g_value : ∃ n, g n = 21 ∧ ∀ m, g m ≤ 21 :=
sorry

end max_g_value_l38_38543


namespace tensor_12_9_l38_38060

def tensor (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem tensor_12_9 : tensor 12 9 = 13 + 7 / 9 :=
by
  sorry

end tensor_12_9_l38_38060


namespace find_x_l38_38517

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 :=
by 
  sorry

end find_x_l38_38517


namespace ratio_s_t_l38_38651

variable {b s t : ℝ}
variable (hb : b ≠ 0)
variable (h1 : s = -b / 8)
variable (h2 : t = -b / 4)

theorem ratio_s_t : s / t = 1 / 2 :=
by
  sorry

end ratio_s_t_l38_38651


namespace polynomial_division_l38_38820

noncomputable def polynomial_div_quotient (p q : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.divByMonic p q

theorem polynomial_division 
  (p q : Polynomial ℚ)
  (hq : q = Polynomial.C 3 * Polynomial.X - Polynomial.C 4)
  (hp : p = 10 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 8 * Polynomial.X - 9) :
  polynomial_div_quotient p q = (10 / 3) * Polynomial.X ^ 2 - (55 / 9) * Polynomial.X - (172 / 27) :=
by
  sorry

end polynomial_division_l38_38820


namespace gain_in_transaction_per_year_l38_38757

noncomputable def compounded_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_per_year (P : ℝ) (t : ℝ) (r1 : ℝ) (n1 : ℕ) (r2 : ℝ) (n2 : ℕ) : ℝ :=
  let amount_repaid := compounded_interest P r1 n1 t
  let amount_received := compounded_interest P r2 n2 t
  (amount_received - amount_repaid) / t

theorem gain_in_transaction_per_year :
  let P := 8000
  let t := 3
  let r1 := 0.05
  let n1 := 2
  let r2 := 0.07
  let n2 := 4
  abs (gain_per_year P t r1 n1 r2 n2 - 191.96) < 0.01 :=
by
  sorry

end gain_in_transaction_per_year_l38_38757


namespace least_values_3198_l38_38315

theorem least_values_3198 (x y : ℕ) (hX : ∃ n : ℕ, 3198 + n * 9 = 27)
                         (hY : ∃ m : ℕ, 3198 + m * 11 = 11) :
  x = 6 ∧ y = 8 :=
by
  sorry

end least_values_3198_l38_38315


namespace find_y_l38_38548

theorem find_y (y : ℝ) (h : (y^2 - 11 * y + 24) / (y - 1) + (4 * y^2 + 20 * y - 25) / (4*y - 5) = 5) :
  y = 3 ∨ y = 4 :=
sorry

end find_y_l38_38548


namespace final_solution_percentage_l38_38647

variable (initial_volume replaced_fraction : ℝ)
variable (initial_concentration replaced_concentration : ℝ)

noncomputable
def final_acid_percentage (initial_volume replaced_fraction initial_concentration replaced_concentration : ℝ) : ℝ :=
  let remaining_volume := initial_volume * (1 - replaced_fraction)
  let replaced_volume := initial_volume * replaced_fraction
  let remaining_acid := remaining_volume * initial_concentration
  let replaced_acid := replaced_volume * replaced_concentration
  let total_acid := remaining_acid + replaced_acid
  let final_volume := initial_volume
  (total_acid / final_volume) * 100

theorem final_solution_percentage :
  final_acid_percentage 100 0.5 0.5 0.3 = 40 :=
by
  sorry

end final_solution_percentage_l38_38647


namespace find_abc_sum_l38_38951

noncomputable def x := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)

theorem find_abc_sum :
  ∃ (a b c : ℕ), a + b + c = 5824 ∧
  x ^ 100 = 3 * x ^ 98 + 15 * x ^ 96 + 12 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40 :=
  sorry

end find_abc_sum_l38_38951


namespace gardener_trees_problem_l38_38785

theorem gardener_trees_problem 
  (maple_trees : ℕ) (oak_trees : ℕ) (birch_trees : ℕ) 
  (total_trees : ℕ) (valid_positions : ℕ) 
  (total_arrangements : ℕ) (probability_numerator : ℕ) (probability_denominator : ℕ) 
  (reduced_numerator : ℕ) (reduced_denominator : ℕ) (m_plus_n : ℕ) :
  (maple_trees = 5) ∧ 
  (oak_trees = 3) ∧ 
  (birch_trees = 7) ∧ 
  (total_trees = 15) ∧ 
  (valid_positions = 8) ∧ 
  (total_arrangements = 120120) ∧ 
  (probability_numerator = 40) ∧ 
  (probability_denominator = total_arrangements) ∧ 
  (reduced_numerator = 1) ∧ 
  (reduced_denominator = 3003) ∧ 
  (m_plus_n = reduced_numerator + reduced_denominator) → 
  m_plus_n = 3004 := 
by
  intros _
  sorry

end gardener_trees_problem_l38_38785


namespace average_time_per_other_class_l38_38874

theorem average_time_per_other_class (school_hours : ℚ) (num_classes : ℕ) (hist_chem_hours : ℚ)
  (total_school_time_minutes : ℕ) (hist_chem_time_minutes : ℕ) (num_other_classes : ℕ)
  (other_classes_time_minutes : ℕ) (average_time_other_classes : ℕ) :
  school_hours = 7.5 →
  num_classes = 7 →
  hist_chem_hours = 1.5 →
  total_school_time_minutes = school_hours * 60 →
  hist_chem_time_minutes = hist_chem_hours * 60 →
  other_classes_time_minutes = total_school_time_minutes - hist_chem_time_minutes →
  num_other_classes = num_classes - 2 →
  average_time_other_classes = other_classes_time_minutes / num_other_classes →
  average_time_other_classes = 72 :=
by
  intros
  sorry

end average_time_per_other_class_l38_38874


namespace u_less_than_v_l38_38073

noncomputable def f (u : ℝ) := (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10 * u^9
noncomputable def g (v : ℝ) := (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10 * v^11

theorem u_less_than_v
  (u v : ℝ)
  (hu : f u = 8)
  (hv : g v = 8) :
  u < v := 
sorry

end u_less_than_v_l38_38073


namespace circles_intersect_l38_38336

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

theorem circles_intersect :
  ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y := by
  sorry

end circles_intersect_l38_38336


namespace cost_of_plastering_l38_38576

def length := 25
def width := 12
def depth := 6
def cost_per_sq_meter_paise := 75

def surface_area_of_two_walls_one := 2 * (length * depth)
def surface_area_of_two_walls_two := 2 * (width * depth)
def surface_area_of_bottom := length * width

def total_surface_area := surface_area_of_two_walls_one + surface_area_of_two_walls_two + surface_area_of_bottom

def cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
def total_cost := total_surface_area * cost_per_sq_meter_rupees

theorem cost_of_plastering : total_cost = 558 := by
  sorry

end cost_of_plastering_l38_38576


namespace probability_of_hitting_exactly_twice_l38_38028

def P_hit_first : ℝ := 0.4
def P_hit_second : ℝ := 0.5
def P_hit_third : ℝ := 0.7

def P_hit_exactly_twice_in_three_shots : ℝ :=
  P_hit_first * P_hit_second * (1 - P_hit_third) +
  (1 - P_hit_first) * P_hit_second * P_hit_third +
  P_hit_first * (1 - P_hit_second) * P_hit_third

theorem probability_of_hitting_exactly_twice :
  P_hit_exactly_twice_in_three_shots = 0.41 := 
by
  sorry

end probability_of_hitting_exactly_twice_l38_38028


namespace least_number_of_stamps_is_6_l38_38711

noncomputable def exist_stamps : Prop :=
∃ (c f : ℕ), 5 * c + 7 * f = 40 ∧ c + f = 6

theorem least_number_of_stamps_is_6 : exist_stamps :=
sorry

end least_number_of_stamps_is_6_l38_38711


namespace sum_of_coordinates_of_other_endpoint_l38_38846

theorem sum_of_coordinates_of_other_endpoint :
  ∀ (x y : ℤ), (7, -15) = ((x + 3) / 2, (y - 5) / 2) → x + y = -14 :=
by
  intros x y h
  sorry

end sum_of_coordinates_of_other_endpoint_l38_38846


namespace mean_value_of_quadrilateral_angles_l38_38962

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l38_38962


namespace four_consecutive_none_multiple_of_5_l38_38873

theorem four_consecutive_none_multiple_of_5 (n : ℤ) :
  (∃ k : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 5 * k) →
  ¬ (∃ m : ℤ, (n = 5 * m) ∨ (n + 1 = 5 * m) ∨ (n + 2 = 5 * m) ∨ (n + 3 = 5 * m)) :=
by sorry

end four_consecutive_none_multiple_of_5_l38_38873


namespace train_speed_l38_38058

noncomputable def trainLength : ℕ := 400
noncomputable def timeToCrossPole : ℕ := 20

theorem train_speed : (trainLength / timeToCrossPole) = 20 := by
  sorry

end train_speed_l38_38058


namespace irene_age_is_46_l38_38570

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l38_38570


namespace problem_l38_38771

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem problem (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1))
                (h2 : f 1 = 0) :
  f 0 = -2 ∧ ∀ x : ℝ, f x = x^2 + x - 2 := by
  sorry

end problem_l38_38771


namespace M_is_correct_l38_38272

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x > 2}

def M := {x | x ∈ A ∧ x ∉ B}

theorem M_is_correct : M = {1, 2} := by
  -- Proof needed here
  sorry

end M_is_correct_l38_38272


namespace equation_solutions_count_l38_38686

theorem equation_solutions_count (n : ℕ) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 2 * x + 3 * y + z + x^2 = n) →
  (n = 32 ∨ n = 33) :=
sorry

end equation_solutions_count_l38_38686


namespace cubic_polynomial_sum_l38_38926

noncomputable def Q (a b c m x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2 * m

theorem cubic_polynomial_sum (a b c m : ℝ) :
  Q a b c m 0 = 2 * m ∧ Q a b c m 1 = 3 * m ∧ Q a b c m (-1) = 5 * m →
  Q a b c m 2 + Q a b c m (-2) = 20 * m :=
by
  intro h
  sorry

end cubic_polynomial_sum_l38_38926


namespace largest_inscribed_triangle_area_l38_38075

-- Definition of the conditions
def radius : ℝ := 10
def diameter : ℝ := 2 * radius

-- The theorem to be proven
theorem largest_inscribed_triangle_area (r : ℝ) (D : ℝ) (h : D = 2 * r) : 
  ∃ (A : ℝ), A = 100 := by
  have base := D
  have height := r
  have area := (1 / 2) * base * height
  use area
  sorry

end largest_inscribed_triangle_area_l38_38075


namespace math_olympiad_problem_l38_38925

theorem math_olympiad_problem (students : Fin 11 → Finset (Fin n)) (h_solved : ∀ i, (students i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → ∃ p, p ∈ students i ∧ p ∉ students j) : 
  6 ≤ n := 
sorry

end math_olympiad_problem_l38_38925


namespace general_formula_for_an_l38_38796

-- Definitions for the first few terms of the sequence
def a1 : ℚ := 1 / 7
def a2 : ℚ := 3 / 77
def a3 : ℚ := 5 / 777

-- The sequence definition as per the identified pattern
def a_n (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

-- The theorem to establish that the sequence definition for general n holds given the initial terms 
theorem general_formula_for_an {n : ℕ} :
  (n = 1 → a_n n = a1) ∧
  (n = 2 → a_n n = a2) ∧ 
  (n = 3 → a_n n = a3) ∧ 
  (∀ n > 3, a_n n = (18 * n - 9) / (7 * (10^n - 1))) := 
by
  sorry

end general_formula_for_an_l38_38796


namespace abby_potatoes_peeled_l38_38884

theorem abby_potatoes_peeled (total_potatoes : ℕ) (homers_rate : ℕ) (abbys_rate : ℕ) (time_alone : ℕ) (potatoes_peeled : ℕ) :
  (total_potatoes = 60) →
  (homers_rate = 4) →
  (abbys_rate = 6) →
  (time_alone = 6) →
  (potatoes_peeled = 22) :=
  sorry

end abby_potatoes_peeled_l38_38884


namespace increase_in_area_l38_38816

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
noncomputable def radius_of_circle (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * (radius ^ 2)

theorem increase_in_area :
  let rectangle_length := 60
  let rectangle_width := 20
  let rectangle_area := area_of_rectangle rectangle_length rectangle_width
  let fence_length := perimeter_of_rectangle rectangle_length rectangle_width
  let circle_radius := radius_of_circle fence_length
  let circle_area := area_of_circle circle_radius
  let area_increase := circle_area - rectangle_area
  837.99 ≤ area_increase :=
by
  sorry

end increase_in_area_l38_38816


namespace greatest_three_digit_divisible_by_3_6_5_l38_38271

/-- Define a three-digit number and conditions for divisibility by 3, 6, and 5 -/
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

/-- Greatest three-digit number divisible by 3, 6, and 5 is 990 -/
theorem greatest_three_digit_divisible_by_3_6_5 : ∃ n : ℕ, is_three_digit n ∧ is_divisible_by n 3 ∧ is_divisible_by n 6 ∧ is_divisible_by n 5 ∧ n = 990 :=
sorry

end greatest_three_digit_divisible_by_3_6_5_l38_38271


namespace sum_first10PrimesGT50_eq_732_l38_38051

def first10PrimesGT50 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

theorem sum_first10PrimesGT50_eq_732 :
  first10PrimesGT50.sum = 732 := by
  sorry

end sum_first10PrimesGT50_eq_732_l38_38051


namespace total_players_l38_38109

-- Definitions of the given conditions
def K : ℕ := 10
def Kho_only : ℕ := 40
def Both : ℕ := 5

-- The lean statement that captures the problem of proving the total number of players equals 50
theorem total_players : (K - Both) + Kho_only + Both = 50 :=
by
  -- Placeholder for the proof
  sorry

end total_players_l38_38109


namespace range_of_a_l38_38158

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- The theorem we need to prove
theorem range_of_a {a : ℝ} (h : A a ⊆ B) : 1 ≤ a ∧ a < 5 := 
sorry

end range_of_a_l38_38158


namespace find_missing_number_l38_38154

theorem find_missing_number (x : ℕ) (h : 10010 - 12 * 3 * x = 9938) : x = 2 :=
by {
  sorry
}

end find_missing_number_l38_38154


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l38_38244

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l38_38244


namespace find_number_l38_38915

theorem find_number (x : ℝ) (h : 0.35 * x = 0.50 * x - 24) : x = 160 :=
by
  sorry

end find_number_l38_38915


namespace box_distribution_l38_38660

theorem box_distribution (A P S : ℕ) (h : A + P + S = 22) : A ≥ 8 ∨ P ≥ 8 ∨ S ≥ 8 := 
by 
-- The next step is to use proof by contradiction, assuming the opposite.
sorry

end box_distribution_l38_38660


namespace unit_cubes_fill_box_l38_38333

theorem unit_cubes_fill_box (p : ℕ) (hp : Nat.Prime p) :
  let length := p
  let width := 2 * p
  let height := 3 * p
  length * width * height = 6 * p^3 :=
by
  -- Proof here
  sorry

end unit_cubes_fill_box_l38_38333


namespace min_value_f_when_a1_l38_38523

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |x - a|

theorem min_value_f_when_a1 : ∀ x : ℝ, f x 1 ≥ 3/4 :=
by sorry

end min_value_f_when_a1_l38_38523


namespace longest_side_of_triangle_l38_38475

theorem longest_side_of_triangle :
  ∀ (A B C a b : ℝ),
    B = 2 * π / 3 →
    C = π / 6 →
    a = 5 →
    A = π - B - C →
    (b / (Real.sin B) = a / (Real.sin A)) →
    b = 5 * Real.sqrt 3 :=
by
  intros A B C a b hB hC ha hA h_sine_ratio
  sorry

end longest_side_of_triangle_l38_38475


namespace a_eq_b_if_fraction_is_integer_l38_38329

theorem a_eq_b_if_fraction_is_integer (a b : ℕ) (h_pos_a : 1 ≤ a) (h_pos_b : 1 ≤ b) :
  ∃ k : ℕ, (a^4 + a^3 + 1) = k * (a^2 * b^2 + a * b^2 + 1) -> a = b :=
by
  sorry

end a_eq_b_if_fraction_is_integer_l38_38329


namespace average_points_per_player_l38_38103

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l38_38103


namespace distinct_sets_count_l38_38901

noncomputable def num_distinct_sets : ℕ :=
  let product : ℕ := 11 * 21 * 31 * 41 * 51 * 61
  728

theorem distinct_sets_count : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11 * 21 * 31 * 41 * 51 * 61 ∧ num_distinct_sets = 728 :=
sorry

end distinct_sets_count_l38_38901


namespace prove_a2_b2_c2_zero_l38_38646

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l38_38646


namespace train_length_72kmphr_9sec_180m_l38_38975

/-- Given speed in km/hr and time in seconds, calculate the length of the train in meters -/
theorem train_length_72kmphr_9sec_180m : ∀ (speed_kmph : ℕ) (time_sec : ℕ),
  speed_kmph = 72 → time_sec = 9 → 
  (speed_kmph * 1000 / 3600) * time_sec = 180 :=
by
  intros speed_kmph time_sec h1 h2
  sorry

end train_length_72kmphr_9sec_180m_l38_38975


namespace geometric_series_first_term_l38_38829

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l38_38829


namespace comic_books_left_l38_38349

theorem comic_books_left (total : ℕ) (sold : ℕ) (left : ℕ) (h1 : total = 90) (h2 : sold = 65) :
  left = total - sold → left = 25 := by
  sorry

end comic_books_left_l38_38349


namespace degree_le_three_l38_38169

theorem degree_le_three
  (d : ℕ)
  (P : Polynomial ℤ)
  (hdeg : P.degree = d)
  (hP : ∃ (S : Finset ℤ), (S.card ≥ d + 1) ∧ ∀ m ∈ S, |P.eval m| = 1) :
  d ≤ 3 := 
sorry

end degree_le_three_l38_38169


namespace equal_real_roots_value_l38_38042

theorem equal_real_roots_value (a c : ℝ) (ha : a ≠ 0) (h : 4 - 4 * a * (2 - c) = 0) : (1 / a) + c = 2 := 
by
  sorry

end equal_real_roots_value_l38_38042


namespace find_ages_l38_38444

theorem find_ages (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 := 
sorry

end find_ages_l38_38444


namespace geometric_sequence_a4_value_l38_38112

theorem geometric_sequence_a4_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h1 : a 1 + (2 / 3) * a 2 = 3) 
  (h2 : a 4^2 = (1 / 9) * a 3 * a 7) 
  :
  a 4 = 27 :=
sorry

end geometric_sequence_a4_value_l38_38112


namespace least_integer_value_l38_38321

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l38_38321


namespace time_to_see_each_other_again_l38_38866

variable (t : ℝ) (t_frac : ℚ)
variable (kenny_speed jenny_speed : ℝ)
variable (kenny_initial jenny_initial : ℝ)
variable (building_side distance_between_paths : ℝ)

def kenny_position (t : ℝ) : ℝ := kenny_initial + kenny_speed * t
def jenny_position (t : ℝ) : ℝ := jenny_initial + jenny_speed * t

theorem time_to_see_each_other_again
  (kenny_speed_eq : kenny_speed = 4)
  (jenny_speed_eq : jenny_speed = 2)
  (kenny_initial_eq : kenny_initial = -50)
  (jenny_initial_eq : jenny_initial = -50)
  (building_side_eq : building_side = 100)
  (distance_between_paths_eq : distance_between_paths = 300)
  (t_gt_50 : t > 50)
  (t_frac_eq : t_frac = 50) :
  (t == t_frac) :=
  sorry

end time_to_see_each_other_again_l38_38866


namespace simultaneous_equations_solution_l38_38611

theorem simultaneous_equations_solution (x y : ℚ) (h1 : 3 * x - 4 * y = 11) (h2 : 9 * x + 6 * y = 33) : 
  x = 11 / 3 ∧ y = 0 :=
by {
  sorry
}

end simultaneous_equations_solution_l38_38611


namespace find_a_and_b_l38_38347

theorem find_a_and_b (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a * b - 9) : a = 3 ∧ b = 3 :=
by
  sorry

end find_a_and_b_l38_38347


namespace percentage_given_away_l38_38903

theorem percentage_given_away
  (initial_bottles : ℕ)
  (drank_percentage : ℝ)
  (remaining_percentage : ℝ)
  (gave_away : ℝ):
  initial_bottles = 3 →
  drank_percentage = 0.90 →
  remaining_percentage = 0.70 →
  gave_away = initial_bottles - (drank_percentage * 1 + remaining_percentage) →
  (gave_away / 2) / 1 * 100 = 70 :=
by
  intros
  sorry

end percentage_given_away_l38_38903


namespace intersection_M_N_l38_38379

def M : Set ℝ := { x | x / (x - 1) ≥ 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_M_N :
  { x | x / (x - 1) ≥ 0 } ∩ { y | ∃ x : ℝ, y = 3 * x^2 + 1 } = { x | x > 1 } :=
sorry

end intersection_M_N_l38_38379


namespace find_R_when_S_is_five_l38_38372

theorem find_R_when_S_is_five (g : ℚ) :
  (∀ (S : ℚ), R = g * S^2 - 5) →
  (R = 25 ∧ S = 3) →
  R = (250 / 3) - 5 :=
by 
  sorry

end find_R_when_S_is_five_l38_38372


namespace function_properties_l38_38278

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

theorem function_properties 
  (b c : ℝ) :
  ((c = 0 → (∀ x : ℝ, f (-x) b 0 = -f x b 0)) ∧
   (b = 0 → (∀ x₁ x₂ : ℝ, (x₁ ≤ x₂ → f x₁ 0 c ≤ f x₂ 0 c))) ∧
   (∃ (c : ℝ), ∀ (x : ℝ), f (x + c) b c = f (x - c) b c) ∧
   (¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0))) := 
by
  sorry

end function_properties_l38_38278


namespace f_odd_function_l38_38429

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (a b : ℝ) : f (a + b) = f a + f b

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

end f_odd_function_l38_38429


namespace inequality_no_real_solutions_l38_38944

theorem inequality_no_real_solutions (a b : ℝ) 
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : 
  |b| ≤ 1 :=
sorry

end inequality_no_real_solutions_l38_38944


namespace decreasing_range_of_a_l38_38661

noncomputable def f (a x : ℝ) : ℝ := (Real.sqrt (2 - a * x)) / (a - 1)

theorem decreasing_range_of_a (a : ℝ) :
    (∀ x y : ℝ, 0 ≤ x → x ≤ 1/2 → 0 ≤ y → y ≤ 1/2 → x < y → f a y < f a x) ↔ (a < 0 ∨ (1 < a ∧ a ≤ 4)) :=
by
  sorry

end decreasing_range_of_a_l38_38661


namespace final_number_is_odd_l38_38147

theorem final_number_is_odd : 
  ∃ (n : ℤ), n % 2 = 1 ∧ n ≥ 1 ∧ n < 1024 := sorry

end final_number_is_odd_l38_38147


namespace four_cubic_feet_to_cubic_inches_l38_38559

theorem four_cubic_feet_to_cubic_inches (h : 1 = 12) : 4 * (12^3) = 6912 :=
by
  sorry

end four_cubic_feet_to_cubic_inches_l38_38559


namespace jake_sister_weight_ratio_l38_38356

theorem jake_sister_weight_ratio
  (jake_present_weight : ℕ)
  (total_weight : ℕ)
  (weight_lost : ℕ)
  (sister_weight : ℕ)
  (jake_weight_after_loss : ℕ)
  (ratio : ℕ) :
  jake_present_weight = 188 →
  total_weight = 278 →
  weight_lost = 8 →
  jake_weight_after_loss = jake_present_weight - weight_lost →
  sister_weight = total_weight - jake_present_weight →
  ratio = jake_weight_after_loss / sister_weight →
  ratio = 2 := by
  sorry

end jake_sister_weight_ratio_l38_38356


namespace geometric_sequence_sum_l38_38439

def a (n : ℕ) : ℕ := 3 * (2 ^ (n - 1))

theorem geometric_sequence_sum :
  a 1 = 3 → a 4 = 24 → (a 3 + a 4 + a 5) = 84 :=
by
  intros h1 h4
  sorry

end geometric_sequence_sum_l38_38439


namespace total_students_in_lunchroom_l38_38993

theorem total_students_in_lunchroom (students_per_table : ℕ) (num_tables : ℕ) (total_students : ℕ) :
  students_per_table = 6 → 
  num_tables = 34 → 
  total_students = students_per_table * num_tables → 
  total_students = 204 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_students_in_lunchroom_l38_38993


namespace polar_to_line_distance_l38_38939

theorem polar_to_line_distance : 
  let point_polar := (2, Real.pi / 3)
  let line_polar := (2, 0)  -- Corresponding (rho, theta) for the given line
  let point_rect := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
  let line_rect := 2  -- x = 2 in rectangular coordinates
  let distance := abs (line_rect - point_rect.1)
  distance = 1 := by
{
  sorry
}

end polar_to_line_distance_l38_38939


namespace arithmetic_sequence_common_difference_l38_38635

theorem arithmetic_sequence_common_difference 
  (a l S : ℕ) (h1 : a = 5) (h2 : l = 50) (h3 : S = 495) :
  (∃ d n : ℕ, l = a + (n-1) * d ∧ S = n * (a + l) / 2 ∧ d = 45 / 17) :=
by
  sorry

end arithmetic_sequence_common_difference_l38_38635


namespace egg_whites_per_cake_l38_38982

-- Define the conversion ratio between tablespoons of aquafaba and egg whites
def tablespoons_per_egg_white : ℕ := 2

-- Define the total amount of aquafaba used for two cakes
def total_tablespoons_for_two_cakes : ℕ := 32

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Prove the number of egg whites needed per cake
theorem egg_whites_per_cake :
  (total_tablespoons_for_two_cakes / tablespoons_per_egg_white) / number_of_cakes = 8 := by
  sorry

end egg_whites_per_cake_l38_38982


namespace find_fx_l38_38342

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = 2 * x^2 + 1) : ∀ x : ℝ, f x = 2 * x - 1 := 
sorry

end find_fx_l38_38342


namespace irrational_sqrt_10_l38_38825

theorem irrational_sqrt_10 : Irrational (Real.sqrt 10) :=
sorry

end irrational_sqrt_10_l38_38825


namespace find_f2_f5_sum_l38_38077

theorem find_f2_f5_sum
  (f : ℤ → ℤ)
  (a b : ℤ)
  (h1 : f 1 = 4)
  (h2 : ∀ z : ℤ, f z = 3 * z + 6)
  (h3 : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y + b) :
  f 2 + f 5 = 33 :=
sorry

end find_f2_f5_sum_l38_38077


namespace min_y_value_l38_38454

noncomputable def y (a x : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_y_value (a : ℝ) (h : a ≠ 0) : 
  (a ≥ 2 → ∃ x, y a x = a^2 - 2) ∧ (a < 2 → ∃ x, y a x = 2*(a-1)^2) :=
sorry

end min_y_value_l38_38454


namespace difference_of_squares_l38_38163

theorem difference_of_squares : 73^2 - 47^2 = 3120 :=
by sorry

end difference_of_squares_l38_38163


namespace real_solutions_equation_l38_38465

theorem real_solutions_equation :
  ∃! x : ℝ, 9 * x^2 - 90 * ⌊ x ⌋ + 99 = 0 :=
sorry

end real_solutions_equation_l38_38465


namespace painting_faces_not_sum_to_nine_l38_38295

def eight_sided_die_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs_that_sum_to_nine : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

theorem painting_faces_not_sum_to_nine :
  let total_pairs := (eight_sided_die_numbers.length * (eight_sided_die_numbers.length - 1)) / 2
  let invalid_pairs := pairs_that_sum_to_nine.length
  total_pairs - invalid_pairs = 24 :=
by
  sorry

end painting_faces_not_sum_to_nine_l38_38295


namespace inequality_system_no_solution_l38_38383

theorem inequality_system_no_solution (k x : ℝ) (h₁ : 1 < x ∧ x ≤ 2) (h₂ : x > k) : k ≥ 2 :=
sorry

end inequality_system_no_solution_l38_38383


namespace least_n_prime_condition_l38_38294

theorem least_n_prime_condition : ∃ n : ℕ, (∀ p : ℕ, Prime p → ¬ Prime (p^2 + n)) ∧ (∀ m : ℕ, 
 (m > 0 ∧ ∀ p : ℕ, Prime p → ¬ Prime (p^2 + m)) → m ≥ 5) ∧ n = 5 := by
  sorry

end least_n_prime_condition_l38_38294


namespace total_cost_eq_4800_l38_38859

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l38_38859


namespace train_passes_tree_in_28_seconds_l38_38632

def km_per_hour_to_meter_per_second (km_per_hour : ℕ) : ℕ :=
  km_per_hour * 1000 / 3600

def pass_tree_time (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  length / (km_per_hour_to_meter_per_second speed_kmh)

theorem train_passes_tree_in_28_seconds :
  pass_tree_time 490 63 = 28 :=
by
  sorry

end train_passes_tree_in_28_seconds_l38_38632


namespace pool_water_after_eight_hours_l38_38948

-- Define the conditions
def hour1_fill_rate := 8
def hour2_and_hour3_fill_rate := 10
def hour4_and_hour5_fill_rate := 14
def hour6_fill_rate := 12
def hour7_fill_rate := 12
def hour8_fill_rate := 12
def hour7_leak := -8
def hour8_leak := -5

-- Calculate the water added in each time period
def water_added := hour1_fill_rate +
                   (hour2_and_hour3_fill_rate * 2) +
                   (hour4_and_hour5_fill_rate * 2) +
                   (hour6_fill_rate + hour7_fill_rate + hour8_fill_rate)

-- Calculate the water lost due to leaks
def water_lost := hour7_leak + hour8_leak  -- Note: Leaks are already negative

-- The final calculation: total water added minus total water lost
def final_water := water_added + water_lost

theorem pool_water_after_eight_hours : final_water = 79 :=
by {
  -- proof steps to check equality are omitted here
  sorry
}

end pool_water_after_eight_hours_l38_38948


namespace largest_vertex_sum_of_parabola_l38_38486

theorem largest_vertex_sum_of_parabola 
  (a T : ℤ)
  (hT : T ≠ 0)
  (h1 : 0 = a * 0^2 + b * 0 + c)
  (h2 : 0 = a * (2 * T) ^ 2 + b * (2 * T) + c)
  (h3 : 36 = a * (2 * T + 2) ^ 2 + b * (2 * T + 2) + c) :
  ∃ N : ℚ, N = -5 / 4 :=
sorry

end largest_vertex_sum_of_parabola_l38_38486


namespace right_triangle_inequality_l38_38055

theorem right_triangle_inequality {a b c : ℝ} (h₁ : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l38_38055


namespace find_integers_l38_38947

theorem find_integers (a b : ℤ) (h1 : a * b = a + b) (h2 : a * b = a - b) : a = 0 ∧ b = 0 :=
by 
  sorry

end find_integers_l38_38947


namespace find_k_such_that_product_minus_one_is_perfect_power_l38_38689

noncomputable def product_of_first_n_primes (n : ℕ) : ℕ :=
  (List.take n (List.filter (Nat.Prime) (List.range n.succ))).prod

theorem find_k_such_that_product_minus_one_is_perfect_power :
  ∀ k : ℕ, ∃ a n : ℕ, (product_of_first_n_primes k) - 1 = a^n ∧ n > 1 ∧ k = 1 :=
by
  sorry

end find_k_such_that_product_minus_one_is_perfect_power_l38_38689


namespace triangle_area_right_angled_l38_38045

theorem triangle_area_right_angled (a : ℝ) (h₁ : 0 < a) (h₂ : a < 24) :
  let b := 24
  let c := 48 - a
  (a^2 + b^2 = c^2) → (1/2) * a * b = 216 :=
by
  sorry

end triangle_area_right_angled_l38_38045


namespace div_equivalence_l38_38653

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l38_38653


namespace nalani_fraction_sold_is_3_over_8_l38_38373

-- Definitions of conditions
def num_dogs : ℕ := 2
def puppies_per_dog : ℕ := 10
def total_amount_received : ℕ := 3000
def price_per_puppy : ℕ := 200

-- Calculation of total puppies and sold puppies
def total_puppies : ℕ := num_dogs * puppies_per_dog
def puppies_sold : ℕ := total_amount_received / price_per_puppy

-- Fraction of puppies sold
def fraction_sold : ℚ := puppies_sold / total_puppies

theorem nalani_fraction_sold_is_3_over_8 :
  fraction_sold = 3 / 8 :=
sorry

end nalani_fraction_sold_is_3_over_8_l38_38373


namespace milk_production_l38_38708

theorem milk_production (a b c d e : ℕ) (f g : ℝ) (hf : f = 0.8) (hg : g = 1.1) :
  ((d : ℝ) * e * g * (b : ℝ) / (a * c)) = 1.1 * b * d * e / (a * c) := by
  sorry

end milk_production_l38_38708


namespace tangent_line_perpendicular_l38_38024

theorem tangent_line_perpendicular (m : ℝ) :
  (∀ x : ℝ, y = 2 * x^2) →
  (∀ x : ℝ, (4 * x - y + m = 0) ∧ (x + 4 * y - 8 = 0) → 
  (16 + 8 * m = 0)) →
  m = -2 :=
by
  sorry

end tangent_line_perpendicular_l38_38024


namespace slope_and_intercept_of_given_function_l38_38004

-- Defining the form of a linear function
def linear_function (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- The given linear function
def given_function (x : ℝ) : ℝ := 3 * x + 2

-- Stating the problem as a theorem
theorem slope_and_intercept_of_given_function :
  (∀ x : ℝ, given_function x = linear_function 3 2 x) :=
by
  intro x
  sorry

end slope_and_intercept_of_given_function_l38_38004


namespace distribute_books_l38_38904

theorem distribute_books (m n : ℕ) (h1 : m = 3*n + 8) (h2 : ∃k, m = 5*k + r ∧ r < 5 ∧ r > 0) : 
  n = 5 ∨ n = 6 :=
by sorry

end distribute_books_l38_38904


namespace melies_meat_purchase_l38_38779

-- Define the relevant variables and conditions
variable (initial_amount : ℕ) (amount_left : ℕ) (cost_per_kg : ℕ)

-- State the main theorem we want to prove
theorem melies_meat_purchase (h1 : initial_amount = 180) (h2 : amount_left = 16) (h3 : cost_per_kg = 82) :
  (initial_amount - amount_left) / cost_per_kg = 2 := by
  sorry

end melies_meat_purchase_l38_38779


namespace determine_k_l38_38502

def f(x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g(x k : ℝ) : ℝ := x^3 - k * x - 10

theorem determine_k : 
  (f (-5) - g (-5) k = -24) → k = 61 := 
by 
-- Begin the proof script here
sorry

end determine_k_l38_38502


namespace probability_of_forming_phrase_l38_38243

theorem probability_of_forming_phrase :
  let cards := ["中", "国", "梦"]
  let n := 6
  let m := 1
  ∃ (p : ℚ), p = (m / n : ℚ) ∧ p = 1 / 6 :=
by
  sorry

end probability_of_forming_phrase_l38_38243


namespace distinct_ordered_pairs_solution_l38_38802

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end distinct_ordered_pairs_solution_l38_38802


namespace inequality_implication_l38_38173

theorem inequality_implication (x : ℝ) : 3 * x + 4 < 5 * x - 6 → x > 5 := 
by {
  sorry
}

end inequality_implication_l38_38173


namespace isosceles_triangle_perimeter_l38_38938

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 6) (h2 : b = 3) (h3 : a ≠ b) :
  (2 * b + a = 15) :=
by
  sorry

end isosceles_triangle_perimeter_l38_38938


namespace time_to_cross_is_30_seconds_l38_38029

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end time_to_cross_is_30_seconds_l38_38029


namespace no_integer_solution_for_system_l38_38970

theorem no_integer_solution_for_system :
  ¬ ∃ (a b c d : ℤ), 
    (a * b * c * d - a = 1961) ∧ 
    (a * b * c * d - b = 961) ∧ 
    (a * b * c * d - c = 61) ∧ 
    (a * b * c * d - d = 1) :=
by {
  sorry
}

end no_integer_solution_for_system_l38_38970


namespace bernoulli_inequality_gt_bernoulli_inequality_lt_l38_38410

theorem bernoulli_inequality_gt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : x > 1 ∨ x < 0) : (1 + h)^x > 1 + h * x := sorry

theorem bernoulli_inequality_lt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : 0 < x) (hx3 : x < 1) : (1 + h)^x < 1 + h * x := sorry

end bernoulli_inequality_gt_bernoulli_inequality_lt_l38_38410


namespace suff_and_necc_l38_38260

variable (x : ℝ)

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem suff_and_necc : (x ∈ (A ∪ B)) ↔ (x ∈ C) := by
  sorry

end suff_and_necc_l38_38260


namespace expected_winnings_l38_38920

def probability_heads : ℚ := 1 / 3
def probability_tails : ℚ := 1 / 2
def probability_edge : ℚ := 1 / 6

def winning_heads : ℚ := 2
def winning_tails : ℚ := 2
def losing_edge : ℚ := -4

def expected_value : ℚ := probability_heads * winning_heads + probability_tails * winning_tails + probability_edge * losing_edge

theorem expected_winnings : expected_value = 1 := by
  sorry

end expected_winnings_l38_38920


namespace number_of_blue_eyed_students_in_k_class_l38_38079

-- Definitions based on the given conditions
def total_students := 40
def blond_hair_to_blue_eyes_ratio := 2.5
def students_with_both := 8
def students_with_neither := 5

-- We need to prove that the number of blue-eyed students is 10
theorem number_of_blue_eyed_students_in_k_class 
  (x : ℕ)  -- number of blue-eyed students
  (H1 : total_students = 40)
  (H2 : ∀ x, blond_hair_to_blue_eyes_ratio * x = number_of_blond_students)
  (H3 : students_with_both = 8)
  (H4 : students_with_neither = 5)
  : x = 10 :=
sorry

end number_of_blue_eyed_students_in_k_class_l38_38079


namespace arithmetic_sequence_a20_l38_38593

theorem arithmetic_sequence_a20 (a : Nat → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l38_38593


namespace true_converses_count_l38_38414

-- Definitions according to the conditions
def parallel_lines (L1 L2 : Prop) : Prop := L1 ↔ L2
def congruent_triangles (T1 T2 : Prop) : Prop := T1 ↔ T2
def vertical_angles (A1 A2 : Prop) : Prop := A1 = A2
def squares_equal (m n : ℝ) : Prop := m = n → (m^2 = n^2)

-- Propositions with their converses
def converse_parallel (L1 L2 : Prop) : Prop := parallel_lines L1 L2 → parallel_lines L2 L1
def converse_congruent (T1 T2 : Prop) : Prop := congruent_triangles T1 T2 → congruent_triangles T2 T1
def converse_vertical (A1 A2 : Prop) : Prop := vertical_angles A1 A2 → vertical_angles A2 A1
def converse_squares (m n : ℝ) : Prop := (m^2 = n^2) → (m = n)

-- Proving the number of true converses
theorem true_converses_count : 
  (∃ L1 L2, converse_parallel L1 L2) →
  (∃ T1 T2, ¬converse_congruent T1 T2) →
  (∃ A1 A2, converse_vertical A1 A2) →
  (∃ m n : ℝ, ¬converse_squares m n) →
  (2 = 2) := by
  intros _ _ _ _
  sorry

end true_converses_count_l38_38414


namespace find_n_l38_38331

theorem find_n (n : ℕ)
  (h1 : ∃ k : ℕ, k = n^3) -- the cube is cut into n^3 unit cubes
  (h2 : ∃ r : ℕ, r = 4 * n^2) -- 4 faces are painted, each with area n^2
  (h3 : 1 / 3 = r / (6 * k)) -- one-third of the total number of faces are red
  : n = 2 :=
by
  sorry

end find_n_l38_38331


namespace find_value_of_pow_function_l38_38777

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_value_of_pow_function :
  (∃ α : ℝ, power_function α 4 = 1/2) →
  ∃ α : ℝ, power_function α (1/4) = 2 :=
by
  sorry

end find_value_of_pow_function_l38_38777


namespace discount_allowed_l38_38001

-- Define the conditions
def CP : ℝ := 100 -- Cost Price (CP) is $100 for simplicity
def MP : ℝ := CP + 0.12 * CP -- Selling price marked 12% above cost price
def Loss : ℝ := 0.01 * CP -- Trader suffers a loss of 1% on CP
def SP : ℝ := CP - Loss -- Selling price after suffering the loss

-- State the equivalent proof problem in Lean
theorem discount_allowed : MP - SP = 13 := by
  sorry

end discount_allowed_l38_38001


namespace sequence_sum_l38_38566

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values used in the problem
def specific_condition (a : ℕ → ℝ) : Prop :=
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450)

-- The proof goal that needs to be established
theorem sequence_sum (a : ℕ → ℝ) (h1 : arithmetic_seq a) (h2 : specific_condition a) : a 2 + a 8 = 180 :=
by
  sorry

end sequence_sum_l38_38566


namespace problem1_problem2_l38_38081

-- Definitions and conditions
def A (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }
def B : Set ℝ := { x | x < -6 ∨ x > 1 }

-- (Ⅰ) Problem statement: Prove that if A ∩ B = ∅, then -6 ≤ m ≤ 0.
theorem problem1 (m : ℝ) : A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 := 
by
  sorry

-- (Ⅱ) Problem statement: Prove that if A ⊆ B, then m < -7 or m > 1.
theorem problem2 (m : ℝ) : A m ⊆ B ↔ m < -7 ∨ m > 1 := 
by
  sorry

end problem1_problem2_l38_38081


namespace arithmetical_puzzle_l38_38424

theorem arithmetical_puzzle (S I X T W E N : ℕ) 
  (h1 : S = 1) 
  (h2 : N % 2 = 0) 
  (h3 : (1 * 100 + I * 10 + X) * 3 = T * 1000 + W * 100 + E * 10 + N) 
  (h4 : ∀ (a b c d e f : ℕ), 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f) :
  T = 5 := sorry

end arithmetical_puzzle_l38_38424


namespace quadratic_other_root_l38_38594

theorem quadratic_other_root (m : ℝ) :
  (2 * 1^2 - m * 1 + 6 = 0) →
  ∃ y : ℝ, y ≠ 1 ∧ (2 * y^2 - m * y + 6 = 0) ∧ (1 * y = 3) :=
by
  intros h
  -- using sorry to skip the actual proof
  sorry

end quadratic_other_root_l38_38594


namespace increase_by_150_percent_l38_38248

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l38_38248


namespace percentage_of_125_equals_75_l38_38766

theorem percentage_of_125_equals_75 (p : ℝ) (h : p * 125 = 75) : p = 60 / 100 :=
by
  sorry

end percentage_of_125_equals_75_l38_38766


namespace cubic_transform_l38_38912

theorem cubic_transform (A B C x z β : ℝ) (h₁ : z = x + β) (h₂ : 3 * β + A = 0) :
  z^3 + A * z^2 + B * z + C = 0 ↔ x^3 + (B - (A^2 / 3)) * x + (C - A * B / 3 + 2 * A^3 / 27) = 0 :=
sorry

end cubic_transform_l38_38912


namespace no_uniformly_colored_rectangle_l38_38557

open Int

def point := (ℤ × ℤ)

def is_green (P : point) : Prop :=
  3 ∣ (P.1 + P.2)

def is_red (P : point) : Prop :=
  ¬ is_green P

def is_rectangle (A B C D : point) : Prop :=
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2

def rectangle_area (A B : point) : ℤ :=
  abs (B.1 - A.1) * abs (B.2 - A.2)

theorem no_uniformly_colored_rectangle :
  ∀ (A B C D : point) (k : ℕ), 
  is_rectangle A B C D →
  rectangle_area A C = 2^k →
  ¬ (is_green A ∧ is_green B ∧ is_green C ∧ is_green D) ∧
  ¬ (is_red A ∧ is_red B ∧ is_red C ∧ is_red D) :=
by sorry

end no_uniformly_colored_rectangle_l38_38557


namespace simple_interest_double_l38_38800

theorem simple_interest_double (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ)
  (h1 : t = 50)
  (h2 : A = 2 * P) 
  (h3 : A - P = P * r * t / 100) :
  r = 2 :=
by
  -- Proof is omitted
  sorry

end simple_interest_double_l38_38800


namespace possible_values_of_N_l38_38131

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l38_38131


namespace base4_division_l38_38916

/-- Given in base 4:
2023_4 div 13_4 = 155_4
We need to prove the quotient is equal to 155_4.
-/
theorem base4_division (n m q r : ℕ) (h1 : n = 2 * 4^3 + 0 * 4^2 + 2 * 4^1 + 3 * 4^0)
    (h2 : m = 1 * 4^1 + 3 * 4^0)
    (h3 : q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0)
    (h4 : n = m * q + r)
    (h5 : 0 ≤ r ∧ r < m):
  q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0 := 
by
  sorry

end base4_division_l38_38916


namespace jess_remaining_blocks_l38_38447

-- Define the number of blocks for each segment of Jess's errands
def blocks_to_post_office : Nat := 24
def blocks_to_store : Nat := 18
def blocks_to_gallery : Nat := 15
def blocks_to_library : Nat := 14
def blocks_to_work : Nat := 22
def blocks_already_walked : Nat := 9

-- Calculate the total blocks to be walked
def total_blocks : Nat :=
  blocks_to_post_office + blocks_to_store + blocks_to_gallery + blocks_to_library + blocks_to_work

-- The remaining blocks Jess needs to walk
def blocks_remaining : Nat :=
  total_blocks - blocks_already_walked

-- The statement to be proved
theorem jess_remaining_blocks : blocks_remaining = 84 :=
by
  sorry

end jess_remaining_blocks_l38_38447


namespace Larry_sessions_per_day_eq_2_l38_38247

variable (x : ℝ)
variable (sessions_per_day_time : ℝ)
variable (feeding_time_per_day : ℝ)
variable (total_time_per_day : ℝ)

theorem Larry_sessions_per_day_eq_2
  (h1: sessions_per_day_time = 30 * x)
  (h2: feeding_time_per_day = 12)
  (h3: total_time_per_day = 72) :
  x = 2 := by
  sorry

end Larry_sessions_per_day_eq_2_l38_38247


namespace perpendicular_lines_implies_m_values_l38_38043

-- Define the equations of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0

-- Define the condition of perpendicularity between lines l1 and l2
def perpendicular (m : ℝ) : Prop :=
  let a1 := (m + 2) / (m - 2)
  let a2 := -3 / m
  a1 * a2 = -1

-- The statement to be proved
theorem perpendicular_lines_implies_m_values (m : ℝ) :
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y → perpendicular m) → (m = -1 ∨ m = 6) :=
sorry

end perpendicular_lines_implies_m_values_l38_38043


namespace find_other_number_l38_38545

/--
Given two numbers A and B, where:
    * The reciprocal of the HCF of A and B is \( \frac{1}{13} \).
    * The reciprocal of the LCM of A and B is \( \frac{1}{312} \).
    * A = 24
Prove that B = 169.
-/
theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 24)
  (h2 : (Nat.gcd A B) = 13)
  (h3 : (Nat.lcm A B) = 312) : 
  B = 169 := 
by 
  sorry

end find_other_number_l38_38545


namespace smallest_square_condition_l38_38883

-- Definition of the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_last_digit_not_zero (n : ℕ) : Prop := n % 10 ≠ 0

def remove_last_two_digits (n : ℕ) : ℕ :=
  n / 100

-- The statement of the theorem we need to prove
theorem smallest_square_condition : 
  ∃ n : ℕ, is_square n ∧ has_last_digit_not_zero n ∧ is_square (remove_last_two_digits n) ∧ 121 ≤ n :=
sorry

end smallest_square_condition_l38_38883


namespace repeating_decimal_to_fraction_l38_38899

theorem repeating_decimal_to_fraction :
  let x := 0.431431431 + 0.000431431431 + 0.000000431431431
  let y := 0.4 + x
  y = 427 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l38_38899


namespace minimum_value_proof_l38_38950

noncomputable def minimum_value (a b c : ℝ) (h : a + b + c = 6) : ℝ :=
  9 / a + 4 / b + 1 / c

theorem minimum_value_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 6) :
  (minimum_value a b c h₃) = 6 :=
sorry

end minimum_value_proof_l38_38950


namespace clare_milk_cartons_l38_38743

def money_given := 47
def cost_per_loaf := 2
def loaves_bought := 4
def cost_per_milk := 2
def money_left := 35

theorem clare_milk_cartons : (money_given - money_left - loaves_bought * cost_per_loaf) / cost_per_milk = 2 :=
by
  sorry

end clare_milk_cartons_l38_38743


namespace product_terms_l38_38971

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

-- a1 = 1 and a10 = 3
axiom geom_seq  (h : ∀ n, a_n (n + 1) = r * a_n n) : a_n 1 = 1 → a_n 10 = 3

theorem product_terms :
  (∀ n, a_n (n + 1) = r * a_n n) → a_n 1 = 1 → a_n 10 = 3 → 
  a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 = 81 :=
by
  intros h1 h2 h3
  sorry

end product_terms_l38_38971


namespace geometric_number_difference_l38_38613

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end geometric_number_difference_l38_38613


namespace necessary_and_sufficient_condition_l38_38310

theorem necessary_and_sufficient_condition (x : ℝ) : (x > 0) ↔ (1 / x > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l38_38310


namespace x_finishes_in_24_days_l38_38118

variable (x y : Type) [Inhabited x] [Inhabited y]

/-- 
  y can finish the work in 16 days,
  y worked for 10 days and left the job,
  x alone needs 9 days to finish the remaining work,
  How many days does x need to finish the work alone?
-/
theorem x_finishes_in_24_days
  (days_y : ℕ := 16)
  (work_done_y : ℕ := 10)
  (work_left_x : ℕ := 9)
  (D_x : ℕ) :
  (1 / days_y : ℚ) * work_done_y + (1 / D_x) * work_left_x = 1 / D_x :=
by
  sorry

end x_finishes_in_24_days_l38_38118


namespace xyz_neg_l38_38981

theorem xyz_neg {a b c x y z : ℝ} 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) 
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 :=
by 
  -- to be proven
  sorry

end xyz_neg_l38_38981


namespace arithmetic_series_product_l38_38292

theorem arithmetic_series_product (a b c : ℝ) (h1 : a = b - d) (h2 : c = b + d) (h3 : a * b * c = 125) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) : b ≥ 5 :=
sorry

end arithmetic_series_product_l38_38292


namespace find_number_of_lines_l38_38602

theorem find_number_of_lines (n : ℕ) (h : (n * (n - 1) / 2) * 8 = 280) : n = 10 :=
by
  sorry

end find_number_of_lines_l38_38602


namespace decimal_equivalence_l38_38025

theorem decimal_equivalence : 4 + 3 / 10 + 9 / 1000 = 4.309 := 
by
  sorry

end decimal_equivalence_l38_38025


namespace sum_a_b_is_nine_l38_38628

theorem sum_a_b_is_nine (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (b + 2 - a)^2 + (a - b)^2 + (b + 2 + a)^2 + (a + b)^2 = 324) 
    (h4 : ∃ a' b', a' = a ∧ b' = b ∧ (b + 2 - a) * 1 = -(b + 2 - a)) : 
  a + b = 9 :=
sorry

end sum_a_b_is_nine_l38_38628


namespace james_two_point_shots_l38_38167

-- Definitions based on conditions
def field_goals := 13
def field_goal_points := 3
def total_points := 79

-- Statement to be proven
theorem james_two_point_shots :
  ∃ x : ℕ, 79 = (field_goals * field_goal_points) + (2 * x) ∧ x = 20 :=
by
  sorry

end james_two_point_shots_l38_38167


namespace problem_l38_38227

theorem problem (w x y z : ℕ) (h : 3^w * 5^x * 7^y * 11^z = 2310) : 3 * w + 5 * x + 7 * y + 11 * z = 26 :=
sorry

end problem_l38_38227


namespace least_number_to_subtract_l38_38476

theorem least_number_to_subtract (x : ℕ) (h : 5026 % 5 = x) : x = 1 :=
by sorry

end least_number_to_subtract_l38_38476


namespace cows_eat_grass_l38_38074

theorem cows_eat_grass (ha_per_cow_per_week : ℝ) (ha_grow_per_week : ℝ) :
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (2, 3, 2, 2) →
    (2 : ℝ) = 3 * 2 * ha_per_cow_per_week - 2 * ha_grow_per_week) → 
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (4, 2, 4, 2) →
    (2 : ℝ) = 2 * 4 * ha_per_cow_per_week - 4 * ha_grow_per_week) → 
  ∃ (cows : ℕ), (6 : ℝ) = cows * 6 * ha_per_cow_per_week - 6 * ha_grow_per_week ∧ cows = 3 :=
sorry

end cows_eat_grass_l38_38074


namespace ratio_of_side_lengths_sum_l38_38441

theorem ratio_of_side_lengths_sum (a b c : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 25) :
  a + b + c = 44 := 
by
  sorry

end ratio_of_side_lengths_sum_l38_38441


namespace solve_real_equation_l38_38749

theorem solve_real_equation (x : ℝ) :
  x^2 * (x + 1)^2 + x^2 = 3 * (x + 1)^2 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
by sorry

end solve_real_equation_l38_38749


namespace solve_for_y_in_equation_l38_38644

theorem solve_for_y_in_equation : ∃ y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 :=
by
  use -4
  sorry

end solve_for_y_in_equation_l38_38644


namespace tens_digit_23_1987_l38_38485

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l38_38485


namespace total_food_amount_l38_38308

-- Define constants for the given problem
def chicken : ℕ := 16
def hamburgers : ℕ := chicken / 2
def hot_dogs : ℕ := hamburgers + 2
def sides : ℕ := hot_dogs / 2

-- Prove the total amount of food Peter will buy is 39 pounds
theorem total_food_amount : chicken + hamburgers + hot_dogs + sides = 39 := by
  sorry

end total_food_amount_l38_38308


namespace sin_theta_plus_pi_over_six_l38_38288

open Real

theorem sin_theta_plus_pi_over_six (theta : ℝ) (h : sin θ + sin (θ + π / 3) = sqrt 3) :
  sin (θ + π / 6) = 1 := 
sorry

end sin_theta_plus_pi_over_six_l38_38288


namespace absolute_value_sum_10_terms_l38_38935

def sequence_sum (n : ℕ) : ℤ := (n^2 - 4 * n + 2)

def term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n - 1)

-- Prove that the sum of the absolute values of the first 10 terms is 66.
theorem absolute_value_sum_10_terms : 
  (|term 1| + |term 2| + |term 3| + |term 4| + |term 5| + 
   |term 6| + |term 7| + |term 8| + |term 9| + |term 10| = 66) := 
by 
  -- Skip the proof
  sorry

end absolute_value_sum_10_terms_l38_38935


namespace value_of_z_l38_38988

theorem value_of_z {x y z : ℤ} (h1 : x = 2) (h2 : y = x^2 - 5) (h3 : z = y^2 - 5) : z = -4 := by
  sorry

end value_of_z_l38_38988


namespace find_weight_l38_38240

-- Define the weight of each box before taking out 20 kg as W
variable (W : ℚ)

-- Define the condition given in the problem
def condition : Prop := 7 * (W - 20) = 3 * W

-- The proof goal is to prove W = 35 under the given condition
theorem find_weight (h : condition W) : W = 35 := by
  sorry

end find_weight_l38_38240


namespace market_price_article_l38_38680

theorem market_price_article (P : ℝ)
  (initial_tax_rate : ℝ := 0.035)
  (reduced_tax_rate : ℝ := 0.033333333333333)
  (difference_in_tax : ℝ := 11) :
  (initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax) → 
  P = 6600 :=
by
  intro h
  /-
  We assume h: initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax
  And we need to show P = 6600.
  The proof steps show that P = 6600 follows logically given h and the provided conditions.
  -/
  sorry

end market_price_article_l38_38680


namespace base7_to_base10_conversion_l38_38300

def convert_base_7_to_10 := 243

namespace Base7toBase10

theorem base7_to_base10_conversion :
  2 * 7^2 + 4 * 7^1 + 3 * 7^0 = 129 := by
  -- The original number 243 in base 7 is expanded and evaluated to base 10.
  sorry

end Base7toBase10

end base7_to_base10_conversion_l38_38300


namespace matchsticks_in_20th_stage_l38_38870

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l38_38870


namespace total_people_museum_l38_38790

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l38_38790


namespace distribution_of_earnings_l38_38304

theorem distribution_of_earnings :
  let payments := [10, 15, 20, 25, 30, 50]
  let total_earnings := payments.sum 
  let equal_share := total_earnings / 6
  50 - equal_share = 25 := by
  sorry

end distribution_of_earnings_l38_38304


namespace problem_f_2019_l38_38289

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f1 : f 1 = 1/4
axiom f2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2019 : f 2019 = -1/2 :=
by
  sorry

end problem_f_2019_l38_38289


namespace shirt_price_l38_38522

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end shirt_price_l38_38522


namespace remainder_7_pow_150_mod_4_l38_38312

theorem remainder_7_pow_150_mod_4 : (7 ^ 150) % 4 = 1 :=
by
  sorry

end remainder_7_pow_150_mod_4_l38_38312


namespace ratio_between_two_numbers_l38_38658

noncomputable def first_number : ℕ := 48
noncomputable def lcm_value : ℕ := 432
noncomputable def second_number : ℕ := 9 * 24  -- Derived from the given conditions in the problem

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

theorem ratio_between_two_numbers 
  (A B : ℕ) 
  (hA : A = first_number) 
  (hLCM : Nat.lcm A B = lcm_value) 
  (hB : B = 9 * 24) : 
  ratio A B = 1 / 4.5 :=
by
  -- Proof would go here
  sorry

end ratio_between_two_numbers_l38_38658


namespace minimum_n_of_colored_balls_l38_38283

theorem minimum_n_of_colored_balls (n : ℕ) (h1 : n ≥ 3)
  (h2 : (n * (n + 1)) / 2 % 10 = 0) : n = 24 :=
sorry

end minimum_n_of_colored_balls_l38_38283


namespace problem_l38_38400

variable {x y : ℝ}

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : (x - y)^2 = 41 := 
by
  sorry

end problem_l38_38400


namespace problem1_l38_38482

theorem problem1 :
  (Real.sqrt (3/2)) * (Real.sqrt (21/4)) / (Real.sqrt (7/2)) = 3/2 :=
sorry

end problem1_l38_38482


namespace decreasing_interval_range_of_a_l38_38629

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem decreasing_interval :
  (∀ x > 0, deriv f x = 1 + log x) →
  { x : ℝ | 0 < x ∧ x < 1/e } = { x | 0 < x ∧ deriv f x < 0 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a * x - 6) →
  a ≤ 5 + log 2 :=
sorry

end decreasing_interval_range_of_a_l38_38629


namespace part1_part2_l38_38929

-- Definitions for part 1
def total_souvenirs := 60
def price_a := 100
def price_b := 60
def total_cost_1 := 4600

-- Definitions for part 2
def max_total_cost := 4500
def twice (m : ℕ) := 2 * m

theorem part1 (x y : ℕ) (hx : x + y = total_souvenirs) (hc : price_a * x + price_b * y = total_cost_1) :
  x = 25 ∧ y = 35 :=
by
  -- You can provide the detailed proof here
  sorry

theorem part2 (m : ℕ) (hm1 : 20 ≤ m) (hm2 : m ≤ 22) (hc2 : price_a * m + price_b * (total_souvenirs - m) ≤ max_total_cost) :
  (m = 20 ∨ m = 21 ∨ m = 22) ∧ 
  ∃ W, W = min (40 * 20 + 3600) (min (40 * 21 + 3600) (40 * 22 + 3600)) ∧ W = 4400 :=
by
  -- You can provide the detailed proof here
  sorry

end part1_part2_l38_38929


namespace cube_root_of_neg_eight_l38_38814

theorem cube_root_of_neg_eight : ∃ x : ℝ, x ^ 3 = -8 ∧ x = -2 := by 
  sorry

end cube_root_of_neg_eight_l38_38814


namespace kenneth_fabric_amount_l38_38986

theorem kenneth_fabric_amount :
  ∃ K : ℤ, (∃ N : ℤ, N = 6 * K ∧ (K * 40 + 140000 = N * 40) ∧ K > 0) ∧ K = 700 :=
by
  sorry

end kenneth_fabric_amount_l38_38986


namespace exponent_fraction_law_l38_38361

theorem exponent_fraction_law :
  (2 ^ 2017 + 2 ^ 2013) / (2 ^ 2017 - 2 ^ 2013) = 17 / 15 :=
  sorry

end exponent_fraction_law_l38_38361


namespace final_movie_ticket_price_l38_38268

variable (initial_price : ℝ) (price_year1 price_year2 price_year3 price_year4 price_year5 : ℝ)

def price_after_years (initial_price : ℝ) : ℝ :=
  let price_year1 := initial_price * 1.12
  let price_year2 := price_year1 * 0.95
  let price_year3 := price_year2 * 1.08
  let price_year4 := price_year3 * 0.96
  let price_year5 := price_year4 * 1.06
  price_year5

theorem final_movie_ticket_price :
  price_after_years 100 = 116.9344512 :=
by
  sorry

end final_movie_ticket_price_l38_38268


namespace family_percentage_eaten_after_dinner_l38_38458

theorem family_percentage_eaten_after_dinner
  (total_brownies : ℕ)
  (children_percentage : ℚ)
  (left_over_brownies : ℕ)
  (lorraine_extra_brownie : ℕ)
  (remaining_percentage : ℚ) :
  total_brownies = 16 →
  children_percentage = 0.25 →
  lorraine_extra_brownie = 1 →
  left_over_brownies = 5 →
  remaining_percentage = 50 := by
  sorry

end family_percentage_eaten_after_dinner_l38_38458


namespace min_moves_move_stack_from_A_to_F_l38_38188

theorem min_moves_move_stack_from_A_to_F : 
  ∀ (squares : Fin 6) (stack : Fin 15), 
  (∃ moves : Nat, 
    (moves >= 0) ∧ 
    (moves == 49) ∧
    ∀ (a b : Fin 6), 
        ∃ (piece_from : Fin 15) (piece_to : Fin 15), 
        ((piece_from > piece_to) → (a ≠ b)) ∧
        (a == 0) ∧ 
        (b == 5)) :=
sorry

end min_moves_move_stack_from_A_to_F_l38_38188


namespace eq_abs_piecewise_l38_38781

theorem eq_abs_piecewise (x : ℝ) : (|x| = if x >= 0 then x else -x) :=
by
  sorry

end eq_abs_piecewise_l38_38781


namespace jelly_ratio_l38_38262

theorem jelly_ratio (G S R P : ℕ) 
  (h1 : G = 2 * S)
  (h2 : R = 2 * P) 
  (h3 : P = 6) 
  (h4 : S = 18) : 
  R / G = 1 / 3 := by
  sorry

end jelly_ratio_l38_38262


namespace books_leftover_l38_38855

theorem books_leftover :
  (1500 * 45) % 47 = 13 :=
by
  sorry

end books_leftover_l38_38855


namespace problem_solution_l38_38066

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x. Prove
    that the number of real solutions to the equation x² - 2⌊x⌋ - 3 = 0 is 3. -/
theorem problem_solution : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^2 - 2 * ⌊x⌋ - 3 = 0 := 
sorry

end problem_solution_l38_38066


namespace water_segment_length_l38_38017

theorem water_segment_length 
  (total_distance : ℝ)
  (find_probability : ℝ)
  (lose_probability : ℝ)
  (probability_equation : total_distance * lose_probability = 750) :
  total_distance = 2500 → 
  find_probability = 7 / 10 →
  lose_probability = 3 / 10 →
  x = 750 :=
by
  intros h1 h2 h3
  sorry

end water_segment_length_l38_38017


namespace find_alpha_l38_38930

theorem find_alpha (α : ℝ) (h : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) :
  α = 13 * Real.pi / 18 :=
sorry

end find_alpha_l38_38930


namespace af_b_lt_bf_a_l38_38519

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem af_b_lt_bf_a (h1 : ∀ x y, 0 < x → 0 < y → x < y → f x > f y)
                    (h2 : ∀ x, 0 < x → f x > 0)
                    (h3 : 0 < a)
                    (h4 : 0 < b)
                    (h5 : a < b) :
  a * f b < b * f a :=
sorry

end af_b_lt_bf_a_l38_38519


namespace emily_has_28_beads_l38_38694

def beads_per_necklace : ℕ := 7
def necklaces : ℕ := 4

def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_has_28_beads : total_beads = 28 := by
  sorry

end emily_has_28_beads_l38_38694


namespace volume_ratio_of_trapezoidal_pyramids_l38_38256

theorem volume_ratio_of_trapezoidal_pyramids 
  (V U : ℝ) (m n m₁ n₁ : ℝ)
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0)
  (h_ratio : U / V = (m₁ + n₁)^2 / (m + n)^2) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 :=
sorry

end volume_ratio_of_trapezoidal_pyramids_l38_38256


namespace carnival_days_l38_38997

-- Define the given conditions
def total_money := 3168
def daily_income := 144

-- Define the main theorem statement
theorem carnival_days : (total_money / daily_income) = 22 := by
  sorry

end carnival_days_l38_38997


namespace fraction_meaningful_iff_l38_38072

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by 
  sorry

end fraction_meaningful_iff_l38_38072


namespace sum_of_binary_digits_of_315_l38_38149

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l38_38149


namespace height_of_stack_of_pots_l38_38446

-- Definitions corresponding to problem conditions
def pot_thickness : ℕ := 1

def top_pot_diameter : ℕ := 16

def bottom_pot_diameter : ℕ := 4

def diameter_decrement : ℕ := 2

-- Number of pots calculation
def num_pots : ℕ := (top_pot_diameter - bottom_pot_diameter) / diameter_decrement + 1

-- The total vertical distance from the bottom of the lowest pot to the top of the highest pot
def total_vertical_distance : ℕ := 
  let inner_heights := num_pots * (top_pot_diameter - pot_thickness + bottom_pot_diameter - pot_thickness) / 2
  let total_thickness := num_pots * pot_thickness
  inner_heights + total_thickness

theorem height_of_stack_of_pots : total_vertical_distance = 65 := 
sorry

end height_of_stack_of_pots_l38_38446


namespace strategy_classification_l38_38008

inductive Player
| A
| B

def A_winning_strategy (n0 : Nat) : Prop :=
  n0 >= 8

def B_winning_strategy (n0 : Nat) : Prop :=
  n0 <= 5

def neither_winning_strategy (n0 : Nat) : Prop :=
  n0 = 6 ∨ n0 = 7

theorem strategy_classification (n0 : Nat) : 
  (A_winning_strategy n0 ∨ B_winning_strategy n0 ∨ neither_winning_strategy n0) := by
    sorry

end strategy_classification_l38_38008


namespace contractor_daily_amount_l38_38917

theorem contractor_daily_amount
  (days_worked : ℕ) (total_days : ℕ) (fine_per_absent_day : ℝ)
  (total_amount : ℝ) (days_absent : ℕ) (amount_received : ℝ) :
  days_worked = total_days - days_absent →
  (total_amount = (days_worked * amount_received - days_absent * fine_per_absent_day)) →
  total_days = 30 →
  fine_per_absent_day = 7.50 →
  total_amount = 685 →
  days_absent = 2 →
  amount_received = 25 :=
by
  sorry

end contractor_daily_amount_l38_38917


namespace A_not_on_transformed_plane_l38_38130

noncomputable def A : ℝ × ℝ × ℝ := (-3, -2, 4)
noncomputable def k : ℝ := -4/5
noncomputable def original_plane (x y z : ℝ) : Prop := 2 * x - 3 * y + z - 5 = 0

noncomputable def transformed_plane (x y z : ℝ) : Prop := 
  2 * x - 3 * y + z + (k * -5) = 0

theorem A_not_on_transformed_plane :
  ¬ transformed_plane (-3) (-2) 4 :=
by
  sorry

end A_not_on_transformed_plane_l38_38130


namespace intercepts_equal_l38_38773

theorem intercepts_equal (a : ℝ) (ha : (a ≠ 0) ∧ (a ≠ 2)) : 
  (a = 1 ∨ a = 2) ↔ (a = 1 ∨ a = 2) := 
by 
  sorry


end intercepts_equal_l38_38773


namespace distance_to_school_l38_38203

variable (v d : ℝ) -- typical speed (v) and distance (d)

theorem distance_to_school :
  (30 / 60 : ℝ) = 1 / 2 ∧ -- 30 minutes is 1/2 hour
  (18 / 60 : ℝ) = 3 / 10 ∧ -- 18 minutes is 3/10 hour
  d = v * (1 / 2) ∧ -- distance for typical day
  d = (v + 12) * (3 / 10) -- distance for quieter day
  → d = 9 := sorry

end distance_to_school_l38_38203


namespace max_property_l38_38382

noncomputable def f : ℚ → ℚ := sorry

axiom f_zero : f 0 = 0
axiom f_pos_of_nonzero : ∀ α : ℚ, α ≠ 0 → f α > 0
axiom f_mul : ∀ α β : ℚ, f (α * β) = f α * f β
axiom f_add : ∀ α β : ℚ, f (α + β) ≤ f α + f β
axiom f_bounded_by_1989 : ∀ m : ℤ, f m ≤ 1989

theorem max_property (α β : ℚ) (h : f α ≠ f β) : f (α + β) = max (f α) (f β) := sorry

end max_property_l38_38382


namespace neg_p_necessary_but_not_sufficient_for_neg_q_l38_38685

variable (p q : Prop)

theorem neg_p_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) : 
  (¬p → ¬q) ∧ (¬q → ¬p) := 
sorry

end neg_p_necessary_but_not_sufficient_for_neg_q_l38_38685


namespace point_symmetric_second_quadrant_l38_38455

theorem point_symmetric_second_quadrant (m : ℝ) 
  (symmetry : ∃ x y : ℝ, P = (-m, m-3) ∧ (-x, -y) = (x, y)) 
  (second_quadrant : ∃ x y : ℝ, P = (-m, m-3) ∧ x < 0 ∧ y > 0) : 
  m < 0 := 
sorry

end point_symmetric_second_quadrant_l38_38455


namespace prob_correct_l38_38231

noncomputable def prob_train_there_when_sam_arrives : ℚ :=
  let total_area := (60 : ℚ) * 60
  let triangle_area := (1 / 2 : ℚ) * 15 * 15
  let parallelogram_area := (30 : ℚ) * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem prob_correct : prob_train_there_when_sam_arrives = 25 / 160 :=
  sorry

end prob_correct_l38_38231


namespace arithmetic_square_root_of_nine_l38_38702

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l38_38702


namespace maximize_squares_l38_38179

theorem maximize_squares (a b : ℕ) (k : ℕ) :
  (a ≠ b) →
  ((∃ (k : ℤ), k ≠ 1 ∧ b = k^2) ↔ 
   (∃ (c₁ c₂ c₃ : ℕ), a * (b + 8) = c₁^2 ∧ b * (a + 8) = c₂^2 ∧ a * b = c₃^2 
     ∧ a = 1)) :=
by { sorry }

end maximize_squares_l38_38179


namespace proof_numbers_exist_l38_38375

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist_l38_38375


namespace quadratic_function_graph_opens_downwards_l38_38280

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

-- The problem statement to prove
theorem quadratic_function_graph_opens_downwards :
  (∀ x : ℝ, (quadratic_function (x + 1) - quadratic_function x) < (quadratic_function x - quadratic_function (x - 1))) :=
by
  -- Proof omitted
  sorry

end quadratic_function_graph_opens_downwards_l38_38280


namespace probability_without_replacement_probability_with_replacement_l38_38434

-- Definition for without replacement context
def without_replacement_total_outcomes : ℕ := 6
def without_replacement_favorable_outcomes : ℕ := 3
def without_replacement_prob : ℚ :=
  without_replacement_favorable_outcomes / without_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers without replacement is 1/2
theorem probability_without_replacement : 
  without_replacement_prob = 1 / 2 := by
  sorry

-- Definition for with replacement context
def with_replacement_total_outcomes : ℕ := 16
def with_replacement_favorable_outcomes : ℕ := 3
def with_replacement_prob : ℚ :=
  with_replacement_favorable_outcomes / with_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers with replacement is 3/16
theorem probability_with_replacement : 
  with_replacement_prob = 3 / 16 := by
  sorry

end probability_without_replacement_probability_with_replacement_l38_38434


namespace basketball_shots_l38_38211

variable (x y : ℕ)

theorem basketball_shots : 3 * x + 2 * y = 26 ∧ x + y = 11 → x = 4 :=
by
  intros h
  sorry

end basketball_shots_l38_38211


namespace sales_price_calculation_l38_38620

variables (C S : ℝ)
def gross_profit := 1.25 * C
def gross_profit_value := 30

theorem sales_price_calculation 
  (h1: gross_profit C = 30) :
  S = 54 :=
sorry

end sales_price_calculation_l38_38620


namespace g_of_minus_1_eq_9_l38_38378

-- defining f(x) and g(f(x)), and stating the objective to prove g(-1)=9
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 5

theorem g_of_minus_1_eq_9 : g (-1) = 9 :=
  sorry

end g_of_minus_1_eq_9_l38_38378


namespace boat_speed_in_still_water_l38_38225

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 13) (h2 : B - S = 9) : B = 11 :=
by
  sorry

end boat_speed_in_still_water_l38_38225


namespace coin_flip_sequences_l38_38687

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l38_38687


namespace find_k_l38_38233

theorem find_k
  (k : ℝ)
  (AB : ℝ × ℝ := (3, 1))
  (AC : ℝ × ℝ := (2, k))
  (BC : ℝ × ℝ := (2 - 3, k - 1))
  (h_perpendicular : AB.1 * BC.1 + AB.2 * BC.2 = 0)
  : k = 4 :=
sorry

end find_k_l38_38233


namespace max_value_of_vector_dot_product_l38_38048

theorem max_value_of_vector_dot_product :
  ∀ (x y : ℝ), (-2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2) → (2 * x - y ≤ 4) :=
by
  intros x y h
  sorry

end max_value_of_vector_dot_product_l38_38048


namespace triangle_no_real_solution_l38_38631

theorem triangle_no_real_solution (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (habc : a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬ (∀ x, x^2 - 2 * b * x + 2 * a * c = 0 ∧
       x^2 - 2 * c * x + 2 * a * b = 0 ∧
       x^2 - 2 * a * x + 2 * b * c = 0) :=
by
  intro H
  sorry

end triangle_no_real_solution_l38_38631


namespace population_ratio_l38_38014

theorem population_ratio
  (P_A P_B P_C P_D P_E P_F : ℕ)
  (h1 : P_A = 8 * P_B)
  (h2 : P_B = 5 * P_C)
  (h3 : P_D = 3 * P_C)
  (h4 : P_D = P_E / 2)
  (h5 : P_F = P_A / 4) :
  P_E / P_B = 6 / 5 := by
    sorry

end population_ratio_l38_38014


namespace circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l38_38963

-- Prove the equation of the circle passing through points A and B with center on a specified line
theorem circle_equation_passing_through_points
  (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (N : ℝ → ℝ → Prop) :
  A = (3, 1) →
  B = (-1, 3) →
  (∀ x y, line x y ↔ 3 * x - y - 2 = 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  sorry :=
sorry

-- Prove the symmetric circle equation regarding a specified line
theorem symmetric_circle_equation
  (N N' : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) :
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, N' x y ↔ (x - 1)^2 + (y - 5)^2 = 10) →
  (∀ x y, line x y ↔ x - y + 3 = 0) →
  sorry :=
sorry

-- Prove the trajectory equation of the midpoint
theorem midpoint_trajectory_equation
  (C : ℝ × ℝ) (N : ℝ → ℝ → Prop) (M_trajectory : ℝ → ℝ → Prop) :
  C = (3, 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, M_trajectory x y ↔ (x - 5 / 2)^2 + (y - 2)^2 = 5 / 2) →
  sorry :=
sorry

end circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l38_38963


namespace domain_ln_l38_38206

theorem domain_ln (x : ℝ) : x^2 - x - 2 > 0 ↔ (x < -1 ∨ x > 2) := by
  sorry

end domain_ln_l38_38206


namespace octagon_area_half_l38_38335

theorem octagon_area_half (parallelogram : ℝ) (h_parallelogram : parallelogram = 1) : 
  (octagon_area : ℝ) =
  1 / 2 := 
  sorry

end octagon_area_half_l38_38335


namespace ratio_of_girls_to_boys_in_biology_class_l38_38705

-- Defining the conditions
def physicsClassStudents : Nat := 200
def biologyClassStudents := physicsClassStudents / 2
def boysInBiologyClass : Nat := 25
def girlsInBiologyClass := biologyClassStudents - boysInBiologyClass

-- Statement of the problem
theorem ratio_of_girls_to_boys_in_biology_class : girlsInBiologyClass / boysInBiologyClass = 3 :=
by
  sorry

end ratio_of_girls_to_boys_in_biology_class_l38_38705


namespace fuel_efficiency_problem_l38_38896

theorem fuel_efficiency_problem :
  let F_highway := 30
  let F_urban := 25
  let F_hill := 20
  let D_highway := 100
  let D_urban := 60
  let D_hill := 40
  let gallons_highway := D_highway / F_highway
  let gallons_urban := D_urban / F_urban
  let gallons_hill := D_hill / F_hill
  let total_gallons := gallons_highway + gallons_urban + gallons_hill
  total_gallons = 7.73 := 
by 
  sorry

end fuel_efficiency_problem_l38_38896


namespace value_of_M_after_subtracting_10_percent_l38_38533

-- Define the given conditions and desired result formally in Lean 4
theorem value_of_M_after_subtracting_10_percent (M : ℝ) (h : 0.25 * M = 0.55 * 2500) :
  M - 0.10 * M = 4950 :=
by
  sorry

end value_of_M_after_subtracting_10_percent_l38_38533


namespace ratio_proof_l38_38023

theorem ratio_proof (X: ℕ) (h: 150 * 2 = 300 * X) : X = 1 := by
  sorry

end ratio_proof_l38_38023


namespace simplify_expression_l38_38187

def is_real (x : ℂ) : Prop := ∃ (y : ℝ), x = y

theorem simplify_expression 
  (x y c : ℝ) 
  (i : ℂ) 
  (hi : i^2 = -1) :
  (x + i*y + c)^2 = (x^2 + c^2 - y^2 + 2 * c * x + (2 * x * y + 2 * c * y) * i) :=
by
  sorry

end simplify_expression_l38_38187


namespace keys_missing_l38_38492

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l38_38492


namespace solve_inequality_l38_38803

noncomputable def inequality (x : ℕ) : Prop :=
  6 * (9 : ℝ)^(1/x) - 13 * (3 : ℝ)^(1/x) * (2 : ℝ)^(1/x) + 6 * (4 : ℝ)^(1/x) ≤ 0

theorem solve_inequality (x : ℕ) (hx : 1 < x) : inequality x ↔ x ≥ 2 :=
by {
  sorry
}

end solve_inequality_l38_38803


namespace x_minus_y_eq_neg_200_l38_38105

theorem x_minus_y_eq_neg_200 (x y : ℤ) (h1 : x + y = 290) (h2 : y = 245) : x - y = -200 := by
  sorry

end x_minus_y_eq_neg_200_l38_38105


namespace sum_of_midpoints_l38_38422

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 10) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 10 :=
by
  sorry

end sum_of_midpoints_l38_38422


namespace ratio_gold_to_green_horses_l38_38581

theorem ratio_gold_to_green_horses (blue_horses purple_horses green_horses gold_horses : ℕ)
    (h1 : blue_horses = 3)
    (h2 : purple_horses = 3 * blue_horses)
    (h3 : green_horses = 2 * purple_horses)
    (h4 : blue_horses + purple_horses + green_horses + gold_horses = 33) :
  gold_horses / gcd gold_horses green_horses = 1 / 6 :=
by
  sorry

end ratio_gold_to_green_horses_l38_38581


namespace sum_even_odd_diff_l38_38266

theorem sum_even_odd_diff (n : ℕ) (h : n = 1500) : 
  let S_odd := n / 2 * (1 + (1 + (n - 1) * 2))
  let S_even := n / 2 * (2 + (2 + (n - 1) * 2))
  (S_even - S_odd) = n :=
by
  sorry

end sum_even_odd_diff_l38_38266


namespace evaluate_expression_l38_38286

theorem evaluate_expression : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31 / 25 :=
by
  sorry

end evaluate_expression_l38_38286


namespace dogs_in_academy_l38_38908

noncomputable def numberOfDogs : ℕ :=
  let allSit := 60
  let allStay := 35
  let allFetch := 40
  let allRollOver := 45
  let sitStay := 20
  let sitFetch := 15
  let sitRollOver := 18
  let stayFetch := 10
  let stayRollOver := 13
  let fetchRollOver := 12
  let sitStayFetch := 11
  let sitStayFetchRoll := 8
  let none := 15
  118 -- final count of dogs in the academy

theorem dogs_in_academy : numberOfDogs = 118 :=
by
  sorry

end dogs_in_academy_l38_38908


namespace particle_max_height_l38_38942

noncomputable def max_height (r ω g : ℝ) : ℝ :=
  (r * ω + g / ω) ^ 2 / (2 * g)

theorem particle_max_height (r ω g : ℝ) (h : ω > Real.sqrt (g / r)) :
    max_height r ω g = (r * ω + g / ω) ^ 2 / (2 * g) :=
sorry

end particle_max_height_l38_38942


namespace Eric_test_score_l38_38044

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end Eric_test_score_l38_38044


namespace tiffany_cans_at_end_of_week_l38_38159

theorem tiffany_cans_at_end_of_week:
  (4 + 2.5 - 1.25 + 0 + 3.75 - 1.5 + 0 = 7.5) :=
by
  sorry

end tiffany_cans_at_end_of_week_l38_38159


namespace tangent_line_at_pi_l38_38148

theorem tangent_line_at_pi :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin x) → 
  ∀ x, x = Real.pi →
  ∀ y, (y = -x + Real.pi) ↔
        (∀ x, y = -x + Real.pi) := 
  sorry

end tangent_line_at_pi_l38_38148


namespace master_efficiency_comparison_l38_38655

theorem master_efficiency_comparison (z_parts : ℕ) (z_hours : ℕ) (l_parts : ℕ) (l_hours : ℕ)
    (hz : z_parts = 5) (hz_time : z_hours = 8)
    (hl : l_parts = 3) (hl_time : l_hours = 4) :
    (z_parts / z_hours : ℚ) < (l_parts / l_hours : ℚ) → false :=
by
  -- This is a placeholder for the proof, which is not needed as per the instructions.
  sorry

end master_efficiency_comparison_l38_38655


namespace total_digits_2500_is_9449_l38_38307

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l38_38307


namespace fundraiser_successful_l38_38339

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l38_38339


namespace solve_system_of_equations_l38_38469

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + 2 * x * y + x^2 - 6 * y - 6 * x + 5 = 0)
  (h2 : y - x + 1 = x^2 - 3 * x) : 
  ((x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) ∨ (x = -2 ∧ y = 7)) ∧ x ≠ 0 ∧ x ≠ 3 :=
by 
  sorry

end solve_system_of_equations_l38_38469


namespace expr_value_l38_38204

variable (x y m n a : ℝ)
variable (hxy : x = -y) (hmn : m * n = 1) (ha : |a| = 3)

theorem expr_value : (a / (m * n) + 2018 * (x + y)) = a := sorry

end expr_value_l38_38204


namespace car_maintenance_fraction_l38_38937

variable (p : ℝ) (f : ℝ)

theorem car_maintenance_fraction (hp : p = 5200)
  (he : p - f * p - (p - 320) = 200) : f = 3 / 130 :=
by
  have hp_pos : p ≠ 0 := by linarith [hp]
  sorry

end car_maintenance_fraction_l38_38937


namespace eccentricity_of_ellipse_l38_38707

noncomputable def eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (l : ℝ → ℝ) (hl : l 0 = 0)
  (h_intersects : ∃ M N : ℝ × ℝ, M ≠ N ∧ (M.1 / a)^2 + (M.2 / b)^2 = 1 ∧ (N.1 / a)^2 + (N.2 / b)^2 = 1 ∧ l M.1 = M.2 ∧ l N.1 = N.2)
  (P : ℝ × ℝ) (hP : (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ P ≠ (0, 0))
  (h_product_slopes : ∀ (Mx Nx Px : ℝ) (k : ℝ),
    l Mx = k * Mx →
    l Nx = k * Nx →
    l Px ≠ k * Px →
    ((k * Mx - P.2) / (Mx - P.1)) * ((k * Nx - P.2) / (Nx - P.1)) = -1/3) :
  eccentricity a b h1 h2 = Real.sqrt (2 / 3) :=
by
  sorry

end eccentricity_of_ellipse_l38_38707


namespace deposit_is_3000_l38_38753

-- Define the constants
def cash_price : ℝ := 8000
def monthly_installment : ℝ := 300
def number_of_installments : ℕ := 30
def savings_by_paying_cash : ℝ := 4000

-- Define the total installment payments
def total_installment_payments : ℝ := number_of_installments * monthly_installment

-- Define the total price paid, which includes the deposit and installments
def total_paid : ℝ := cash_price + savings_by_paying_cash

-- Define the deposit
def deposit : ℝ := total_paid - total_installment_payments

-- Statement to be proven
theorem deposit_is_3000 : deposit = 3000 := 
by 
  sorry

end deposit_is_3000_l38_38753


namespace natural_number_40_times_smaller_l38_38898

-- Define the sum of the first (n-1) natural numbers
def sum_natural_numbers (n : ℕ) := (n * (n - 1)) / 2

-- Define the proof statement
theorem natural_number_40_times_smaller (n : ℕ) (h : sum_natural_numbers n = 40 * n) : n = 81 :=
by {
  -- The proof is left as an exercise
  sorry
}

end natural_number_40_times_smaller_l38_38898


namespace abc_sum_eq_11_sqrt_6_l38_38180

variable {a b c : ℝ}

theorem abc_sum_eq_11_sqrt_6 : 
  0 < a → 0 < b → 0 < c → 
  a * b = 36 → 
  a * c = 72 → 
  b * c = 108 → 
  a + b + c = 11 * Real.sqrt 6 :=
by sorry

end abc_sum_eq_11_sqrt_6_l38_38180


namespace supplementary_angle_l38_38478

theorem supplementary_angle {α β : ℝ} (angle_supplementary : α + β = 180) (angle_1_eq : α = 80) : β = 100 :=
by
  sorry

end supplementary_angle_l38_38478


namespace rectangle_width_of_square_l38_38797

theorem rectangle_width_of_square (side_length_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (h1 : side_length_square = 3) (h2 : length_rectangle = 3)
  (h3 : (side_length_square ^ 2) = length_rectangle * width_rectangle) : width_rectangle = 3 :=
by
  sorry

end rectangle_width_of_square_l38_38797


namespace inequality_solution_l38_38062

theorem inequality_solution {x : ℝ} (h : -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2) : 
  x ∈ Set.Ioo (-2 : ℝ) (10 / 3) :=
by
  sorry

end inequality_solution_l38_38062


namespace theo_drinks_8_cups_per_day_l38_38679

/--
Theo, Mason, and Roxy are siblings. 
Mason drinks 7 cups of water every day.
Roxy drinks 9 cups of water every day. 
In one week, the siblings drink 168 cups of water together. 

Prove that Theo drinks 8 cups of water every day.
-/
theorem theo_drinks_8_cups_per_day (T : ℕ) :
  (∀ (d m r : ℕ), 
    (m = 7 ∧ r = 9 ∧ d + m + r = 168) → 
    (T * 7 = d) → T = 8) :=
by
  intros d m r cond1 cond2
  have h1 : d + 49 + 63 = 168 := by sorry
  have h2 : T * 7 = d := cond2
  have goal : T = 8 := by sorry
  exact goal

end theo_drinks_8_cups_per_day_l38_38679


namespace Sravan_travel_time_l38_38327

theorem Sravan_travel_time :
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 15 :=
by
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  sorry

end Sravan_travel_time_l38_38327


namespace Terry_has_20_more_stickers_than_Steven_l38_38592

theorem Terry_has_20_more_stickers_than_Steven :
  let Ryan_stickers := 30
  let Steven_stickers := 3 * Ryan_stickers
  let Total_stickers := 230
  let Ryan_Steven_Total := Ryan_stickers + Steven_stickers
  let Terry_stickers := Total_stickers - Ryan_Steven_Total
  (Terry_stickers - Steven_stickers) = 20 := 
by 
  sorry

end Terry_has_20_more_stickers_than_Steven_l38_38592


namespace rectangular_plot_breadth_l38_38212

theorem rectangular_plot_breadth (b l : ℝ) (A : ℝ)
  (h1 : l = 3 * b)
  (h2 : A = l * b)
  (h3 : A = 2700) : b = 30 :=
by sorry

end rectangular_plot_breadth_l38_38212


namespace minimum_value_of_weighted_sum_l38_38021

theorem minimum_value_of_weighted_sum 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) :
  3 * a + 6 * b + 9 * c ≥ 54 :=
sorry

end minimum_value_of_weighted_sum_l38_38021


namespace quadratic_relationship_l38_38436

theorem quadratic_relationship :
  ∀ (x z : ℕ), (x = 1 ∧ z = 5) ∨ (x = 2 ∧ z = 12) ∨ (x = 3 ∧ z = 23) ∨ (x = 4 ∧ z = 38) ∨ (x = 5 ∧ z = 57) →
  z = 2 * x^2 + x + 2 :=
by
  sorry

end quadratic_relationship_l38_38436


namespace arithmetic_example_l38_38806

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l38_38806


namespace steven_erasers_l38_38890

theorem steven_erasers (skittles erasers groups items_per_group total_items : ℕ)
  (h1 : skittles = 4502)
  (h2 : groups = 154)
  (h3 : items_per_group = 57)
  (h4 : total_items = groups * items_per_group)
  (h5 : total_items - skittles = erasers) :
  erasers = 4276 :=
by
  sorry

end steven_erasers_l38_38890


namespace only_other_list_with_same_product_l38_38255

-- Assigning values to letters
def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7 | 'H' => 8
  | 'I' => 9 | 'J' => 10| 'K' => 11| 'L' => 12| 'M' => 13| 'N' => 14| 'O' => 15| 'P' => 16
  | 'Q' => 17| 'R' => 18| 'S' => 19| 'T' => 20| 'U' => 21| 'V' => 22| 'W' => 23| 'X' => 24
  | 'Y' => 25| 'Z' => 26| _ => 0

-- Define the product function for a list of 4 letters
def product_of_list (lst : List Char) : ℕ :=
  lst.map letter_value |> List.prod

-- Define the specific lists
def BDFH : List Char := ['B', 'D', 'F', 'H']
def BCDH : List Char := ['B', 'C', 'D', 'H']

-- The main statement to prove
theorem only_other_list_with_same_product : 
  product_of_list BCDH = product_of_list BDFH :=
by
  -- Sorry is a placeholder for the proof
  sorry

end only_other_list_with_same_product_l38_38255


namespace system_inequalities_1_l38_38872

theorem system_inequalities_1 (x : ℝ) (h1 : 2 * x ≥ x - 1) (h2 : 4 * x + 10 > x + 1) :
  x ≥ -1 :=
sorry

end system_inequalities_1_l38_38872


namespace expressions_divisible_by_17_l38_38999

theorem expressions_divisible_by_17 (a b : ℤ) : 
  let x := 3 * b - 5 * a
  let y := 9 * a - 2 * b
  (∃ k : ℤ, (2 * x + 3 * y) = 17 * k) ∧ (∃ k : ℤ, (9 * x + 5 * y) = 17 * k) :=
by
  exact ⟨⟨a, by sorry⟩, ⟨b, by sorry⟩⟩

end expressions_divisible_by_17_l38_38999


namespace find_incomes_l38_38101

theorem find_incomes (M N O P Q : ℝ) 
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (O + P) / 2 = 6800)
  (h4 : (P + Q) / 2 = 7500)
  (h5 : (M + O + Q) / 3 = 6000) :
  M = 300 ∧ N = 9800 ∧ O = 2700 ∧ P = 10900 ∧ Q = 4100 :=
by
  sorry


end find_incomes_l38_38101


namespace problem_part_I_problem_part_II_l38_38137

-- Problem (I)
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : 
  a + b = 2 * c -> (a + b) = 2 * c :=
by
  intros h
  sorry

-- Problem (II)
theorem problem_part_II (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = Real.pi / 3) 
  (h2 : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) 
  (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
  (h4 : a + b = 2 * c) : c = 4 :=
by
  intros
  sorry

end problem_part_I_problem_part_II_l38_38137


namespace radius_of_cone_l38_38726

theorem radius_of_cone (S : ℝ) (h_S: S = 9 * Real.pi) (h_net: net_is_semi_circle) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_cone_l38_38726


namespace find_k_l38_38316

theorem find_k (k : ℚ) :
  (∃ (x y : ℚ), y = 4 * x + 5 ∧ y = -3 * x + 10 ∧ y = 2 * x + k) →
  k = 45 / 7 :=
by
  sorry

end find_k_l38_38316


namespace arithmetic_sequence_inequality_l38_38156

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality 
  (h : is_arithmetic_sequence a d)
  (d_pos : d ≠ 0)
  (a_pos : ∀ n, a n > 0) :
  (a 1) * (a 8) < (a 4) * (a 5) := 
by
  sorry

end arithmetic_sequence_inequality_l38_38156


namespace num_zeros_in_interval_l38_38564

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + 7

theorem num_zeros_in_interval : 
    (∃ (a b : ℝ), a < b ∧ a = 0 ∧ b = 2 ∧
     (∀ x, f x = 0 → (0 < x ∧ x < 2)) ∧
     (∃! x, (0 < x ∧ x < 2) ∧ f x = 0)) :=
by
    sorry

end num_zeros_in_interval_l38_38564


namespace base_k_addition_is_ten_l38_38868

theorem base_k_addition_is_ten :
  ∃ k : ℕ, (k > 4) ∧ (5 * k^3 + 3 * k^2 + 4 * k + 2 + 6 * k^3 + 4 * k^2 + 2 * k + 1 = 1 * k^4 + 4 * k^3 + 1 * k^2 + 6 * k + 3) ∧ k = 10 :=
by
  sorry

end base_k_addition_is_ten_l38_38868


namespace sandra_stickers_l38_38681

theorem sandra_stickers :
  ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ (N % 11 = 1) ∧ N = 166 :=
by {
  sorry
}

end sandra_stickers_l38_38681


namespace inhabitant_eq_resident_l38_38750

-- Definitions
def inhabitant : Type := String
def resident : Type := String

-- The equivalence theorem
theorem inhabitant_eq_resident :
  ∀ (x : inhabitant), x = "resident" :=
by
  sorry

end inhabitant_eq_resident_l38_38750


namespace problem_statement_l38_38005

variable (f : ℕ → ℝ)

theorem problem_statement (hf : ∀ k : ℕ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)
  (h : f 4 = 25) : ∀ k : ℕ, k ≥ 4 → f k ≥ k^2 := 
by
  sorry

end problem_statement_l38_38005


namespace value_of_a_l38_38539

theorem value_of_a
  (a b : ℚ)
  (h1 : b / a = 4)
  (h2 : b = 18 - 6 * a) :
  a = 9 / 5 := by
  sorry

end value_of_a_l38_38539


namespace ellipse_properties_l38_38991

theorem ellipse_properties :
  ∀ {x y : ℝ}, 4 * x^2 + 2 * y^2 = 16 →
    (∃ a b e c, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ e = Real.sqrt 2 / 2 ∧ c = 2) ∧
    (∃ f1 f2, f1 = (0, 2) ∧ f2 = (0, -2)) ∧
    (∃ v1 v2 v3 v4, v1 = (0, 2 * Real.sqrt 2) ∧ v2 = (0, -2 * Real.sqrt 2) ∧ v3 = (2, 0) ∧ v4 = (-2, 0)) :=
by
  sorry

end ellipse_properties_l38_38991


namespace route_difference_l38_38782

noncomputable def time_route_A (distance_A : ℝ) (speed_A : ℝ) : ℝ :=
  (distance_A / speed_A) * 60

noncomputable def time_route_B (distance1_B distance2_B distance3_B : ℝ) (speed1_B speed2_B speed3_B : ℝ) : ℝ :=
  ((distance1_B / speed1_B) * 60) + 
  ((distance2_B / speed2_B) * 60) + 
  ((distance3_B / speed3_B) * 60)

theorem route_difference
  (distance_A : ℝ := 8)
  (speed_A : ℝ := 25)
  (distance1_B : ℝ := 2)
  (distance2_B : ℝ := 0.5)
  (speed1_B : ℝ := 50)
  (speed2_B : ℝ := 20)
  (distance_total_B : ℝ := 7)
  (speed3_B : ℝ := 35) :
  time_route_A distance_A speed_A - time_route_B distance1_B distance2_B (distance_total_B - distance1_B - distance2_B) speed1_B speed2_B speed3_B = 7.586 :=
by
  sorry

end route_difference_l38_38782


namespace prove_parabola_points_l38_38643

open Real

noncomputable def parabola_equation (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def dist_to_focus (x y focus_x focus_y : ℝ) : ℝ :=
  (sqrt ((x - focus_x)^2 + (y - focus_y)^2))

theorem prove_parabola_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  parabola_equation x1 y1 →
  parabola_equation x2 y2 →
  dist_to_focus x1 y1 0 1 - dist_to_focus x2 y2 0 1 = 2 →
  (y1 + x1^2 - y2 - x2^2 = 10) :=
by
  intros x1 y1 x2 y2 h₁ h₂ h₃
  sorry

end prove_parabola_points_l38_38643


namespace fraction_of_5100_l38_38460

theorem fraction_of_5100 (x : ℝ) (h : ((3 / 4) * x * (2 / 5) * 5100 = 765.0000000000001)) : x = 0.5 :=
by
  sorry

end fraction_of_5100_l38_38460


namespace inequality_holds_for_all_x_iff_a_in_interval_l38_38111

theorem inequality_holds_for_all_x_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end inequality_holds_for_all_x_iff_a_in_interval_l38_38111


namespace chromium_percentage_new_alloy_l38_38089

variable (w1 w2 : ℝ) (cr1 cr2 : ℝ)

theorem chromium_percentage_new_alloy (h_w1 : w1 = 15) (h_w2 : w2 = 30) (h_cr1 : cr1 = 0.12) (h_cr2 : cr2 = 0.08) :
  (cr1 * w1 + cr2 * w2) / (w1 + w2) * 100 = 9.33 := by
  sorry

end chromium_percentage_new_alloy_l38_38089


namespace determine_q_l38_38715

theorem determine_q (p q : ℝ) 
  (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q * x + 12) : 
  q = 7 :=
by
  sorry

end determine_q_l38_38715


namespace work_days_l38_38577

theorem work_days (hp : ℝ) (hq : ℝ) (fraction_left : ℝ) (d : ℝ) :
  hp = 1 / 20 → hq = 1 / 10 → fraction_left = 0.7 → (3 / 20) * d = (1 - fraction_left) → d = 2 :=
  by
  intros hp_def hq_def fraction_def work_eq
  sorry

end work_days_l38_38577


namespace acute_triangle_and_angle_relations_l38_38789

theorem acute_triangle_and_angle_relations (a b c u v w : ℝ) (A B C : ℝ)
  (h₁ : a^2 = u * (v + w - u))
  (h₂ : b^2 = v * (w + u - v))
  (h₃ : c^2 = w * (u + v - w)) :
  (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
  (∀ U V W : ℝ, U = 180 - 2 * A ∧ V = 180 - 2 * B ∧ W = 180 - 2 * C) :=
by sorry

end acute_triangle_and_angle_relations_l38_38789


namespace linear_function_quadrants_l38_38531

theorem linear_function_quadrants : 
  ∀ (x y : ℝ), y = -5 * x + 3 
  → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by 
  intro x y h
  sorry

end linear_function_quadrants_l38_38531


namespace apples_in_pile_l38_38894

-- Define the initial number of apples in the pile
def initial_apples : ℕ := 8

-- Define the number of added apples
def added_apples : ℕ := 5

-- Define the total number of apples
def total_apples : ℕ := initial_apples + added_apples

-- Prove that the total number of apples is 13
theorem apples_in_pile : total_apples = 13 :=
by
  sorry

end apples_in_pile_l38_38894


namespace pizza_toppings_l38_38168

theorem pizza_toppings :
  ∀ (F V T : ℕ), F = 4 → V = 16 → F * (1 + T) = V → T = 3 :=
by
  intros F V T hF hV h
  sorry

end pizza_toppings_l38_38168


namespace locus_area_l38_38512

theorem locus_area (R : ℝ) (r : ℝ) (hR : R = 6 * Real.sqrt 7) (hr : r = Real.sqrt 7) :
    ∃ (L : ℝ), (L = 2 * Real.sqrt 42 ∧ L^2 * Real.pi = 168 * Real.pi) :=
by
  sorry

end locus_area_l38_38512


namespace inequality_proof_l38_38194

variable (a : ℝ)

theorem inequality_proof (a : ℝ) : 
  (a^2 + a + 2) / (Real.sqrt (a^2 + a + 1)) ≥ 2 :=
sorry

end inequality_proof_l38_38194


namespace principal_sum_l38_38417

theorem principal_sum (A1 A2 : ℝ) (I P : ℝ) 
  (hA1 : A1 = 1717) 
  (hA2 : A2 = 1734) 
  (hI : I = A2 - A1)
  (h_simple_interest : A1 = P + I) : P = 1700 :=
by
  sorry

end principal_sum_l38_38417


namespace arithmetic_sequence_a7_l38_38431

theorem arithmetic_sequence_a7 (S_13 : ℕ → ℕ → ℕ) (n : ℕ) (a7 : ℕ) (h1: S_13 13 52 = 52) (h2: S_13 13 a7 = 13 * a7):
  a7 = 4 :=
by
  sorry

end arithmetic_sequence_a7_l38_38431


namespace factorization_of_w4_minus_81_l38_38959

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l38_38959


namespace total_trip_duration_proof_l38_38719

-- Naming all components
def driving_time : ℝ := 5
def first_jam_duration (pre_first_jam_drive : ℝ) : ℝ := 1.5 * pre_first_jam_drive
def second_jam_duration (between_first_and_second_drive : ℝ) : ℝ := 2 * between_first_and_second_drive
def third_jam_duration (between_second_and_third_drive : ℝ) : ℝ := 3 * between_second_and_third_drive
def pit_stop_duration : ℝ := 0.5
def pit_stops : ℕ := 2
def initial_drive : ℝ := 1
def second_drive : ℝ := 1.5

-- Additional drive time calculation
def remaining_drive : ℝ := driving_time - initial_drive - second_drive

-- Total duration calculation
def total_duration (initial_drive : ℝ) (second_drive : ℝ) (remaining_drive : ℝ) (first_jam_duration : ℝ) 
(second_jam_duration : ℝ) (third_jam_duration : ℝ) (pit_stop_duration : ℝ) (pit_stops : ℕ) : ℝ :=
  driving_time + first_jam_duration + second_jam_duration + third_jam_duration + (pit_stop_duration * pit_stops)

theorem total_trip_duration_proof :
  total_duration initial_drive second_drive remaining_drive (first_jam_duration initial_drive)
                  (second_jam_duration second_drive) (third_jam_duration remaining_drive) pit_stop_duration pit_stops 
  = 18 :=
by
  -- Proof steps would go here
  sorry

end total_trip_duration_proof_l38_38719


namespace John_leftover_money_l38_38151

variables (q : ℝ)

def drinks_price (q : ℝ) : ℝ := 4 * q
def small_pizza_price (q : ℝ) : ℝ := q
def large_pizza_price (q : ℝ) : ℝ := 4 * q
def total_cost (q : ℝ) : ℝ := drinks_price q + small_pizza_price q + 2 * large_pizza_price q
def John_initial_money : ℝ := 50
def John_money_left (q : ℝ) : ℝ := John_initial_money - total_cost q

theorem John_leftover_money : John_money_left q = 50 - 13 * q :=
by
  sorry

end John_leftover_money_l38_38151


namespace fraction_cubed_sum_l38_38389

theorem fraction_cubed_sum (x y : ℤ) (h1 : x = 3) (h2 : y = 4) :
  (x^3 + 3 * y^3) / 7 = 31 + 3 / 7 := by
  sorry

end fraction_cubed_sum_l38_38389


namespace correct_transformation_l38_38695

theorem correct_transformation (x : ℝ) (h : 3 * x - 7 = 2 * x) : 3 * x - 2 * x = 7 :=
sorry

end correct_transformation_l38_38695


namespace tan_identity_l38_38250

theorem tan_identity
  (α : ℝ)
  (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := 
sorry

end tan_identity_l38_38250


namespace find_six_y_minus_four_squared_l38_38673

theorem find_six_y_minus_four_squared (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) :
  (6 * y - 4)^2 = 134 :=
by
  sorry

end find_six_y_minus_four_squared_l38_38673


namespace centroid_quad_area_correct_l38_38817

noncomputable def centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  if h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35 then
    12800 / 9
  else
    sorry

theorem centroid_quad_area_correct (E F G H Q : ℝ × ℝ) (side_length EQ FQ : ℝ) 
  (h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35) :
  centroid_quadrilateral_area E F G H Q side_length EQ FQ = 12800 / 9 :=
sorry

end centroid_quad_area_correct_l38_38817


namespace negation_of_exists_l38_38425

theorem negation_of_exists (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x < 0) ↔ ¬ (∀ x : ℝ, x^2 + 2 * x >= 0) :=
sorry

end negation_of_exists_l38_38425


namespace green_beans_to_onions_ratio_l38_38848

def cut_conditions
  (potatoes : ℕ)
  (carrots : ℕ)
  (onions : ℕ)
  (green_beans : ℕ) : Prop :=
  carrots = 6 * potatoes ∧ onions = 2 * carrots ∧ potatoes = 2 ∧ green_beans = 8

theorem green_beans_to_onions_ratio (potatoes carrots onions green_beans : ℕ) :
  cut_conditions potatoes carrots onions green_beans →
  green_beans / gcd green_beans onions = 1 ∧ onions / gcd green_beans onions = 3 :=
by
  sorry

end green_beans_to_onions_ratio_l38_38848


namespace Q_coordinates_l38_38115

def P : (ℝ × ℝ) := (2, -6)

def Q (x : ℝ) : (ℝ × ℝ) := (x, -6)

axiom PQ_parallel_to_x_axis : ∀ x, Q x = (x, -6)

axiom PQ_length : dist (Q 0) P = 2 ∨ dist (Q 4) P = 2

theorem Q_coordinates : Q 0 = (0, -6) ∨ Q 4 = (4, -6) :=
by {
  sorry
}

end Q_coordinates_l38_38115


namespace team_A_days_additional_people_l38_38843

theorem team_A_days (x : ℕ) (y : ℕ)
  (h1 : 2700 / x = 2 * (1800 / y))
  (h2 : y = x + 1)
  : x = 3 ∧ y = 4 :=
by
  sorry

theorem additional_people (m : ℕ)
  (h1 : (200 : ℝ) * 10 * 3 + 150 * 8 * 4 = 10800)
  (h2 : (170 : ℝ) * (10 + m) * 3 + 150 * 8 * 4 = 1.20 * 10800)
  : m = 6 :=
by
  sorry

end team_A_days_additional_people_l38_38843


namespace initial_bottle_caps_l38_38126

theorem initial_bottle_caps (end_caps : ℕ) (eaten_caps : ℕ) (start_caps : ℕ) 
  (h1 : end_caps = 61) 
  (h2 : eaten_caps = 4) 
  (h3 : start_caps = end_caps + eaten_caps) : 
  start_caps = 65 := 
by 
  sorry

end initial_bottle_caps_l38_38126


namespace range_of_c_l38_38481

def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (1 - 2 * c < 0)

theorem range_of_c (c : ℝ) : (p c ∨ q c) ∧ ¬ (p c ∧ q c) ↔ (0 < c ∧ c ≤ 1/2) ∨ (1 < c) :=
by sorry

end range_of_c_l38_38481


namespace intersection_A_B_subset_A_B_l38_38815

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
noncomputable def set_B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 22}

theorem intersection_A_B (a : ℝ) (ha : a = 10) : set_A a ∩ set_B = {x : ℝ | 21 ≤ x ∧ x ≤ 22} := by
  sorry

theorem subset_A_B (a : ℝ) : set_A a ⊆ set_B → a ≤ 9 := by
  sorry

end intersection_A_B_subset_A_B_l38_38815


namespace solution_set_of_f_x_gt_2_minimum_value_of_f_l38_38964

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solution_set_of_f_x_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7} ∪ {x : ℝ | x > 5 / 3} :=
by 
  sorry

theorem minimum_value_of_f : ∃ x : ℝ, f x = -9 / 2 :=
by 
  sorry

end solution_set_of_f_x_gt_2_minimum_value_of_f_l38_38964


namespace find_three_digit_number_l38_38116

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end find_three_digit_number_l38_38116


namespace marks_lost_per_wrong_answer_l38_38738

theorem marks_lost_per_wrong_answer (x : ℝ) : 
  (score_per_correct = 4) ∧ 
  (num_questions = 60) ∧ 
  (total_marks = 120) ∧ 
  (correct_answers = 36) ∧ 
  (wrong_answers = num_questions - correct_answers) ∧
  (wrong_answers = 24) ∧
  (total_score_from_correct = score_per_correct * correct_answers) ∧ 
  (total_marks_lost = total_score_from_correct - total_marks) ∧ 
  (total_marks_lost = wrong_answers * x) → 
  x = 1 := 
by 
  sorry

end marks_lost_per_wrong_answer_l38_38738


namespace value_sq_dist_OP_OQ_l38_38563

-- Definitions from problem conditions
def origin : ℝ × ℝ := (0, 0)
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def perpendicular (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

-- The proof statement
theorem value_sq_dist_OP_OQ 
  (P Q : ℝ × ℝ) 
  (hP : ellipse P.1 P.2) 
  (hQ : ellipse Q.1 Q.2) 
  (h_perp : perpendicular P Q)
  : (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) = 48 / 7 := 
sorry

end value_sq_dist_OP_OQ_l38_38563


namespace number_of_distinct_values_l38_38955

theorem number_of_distinct_values (n : ℕ) (mode_count : ℕ) (second_count : ℕ) (total_count : ℕ) 
    (h1 : n = 3000) (h2 : mode_count = 15) (h3 : second_count = 14) : 
    (n - mode_count - second_count) / 13 + 2 ≥ 232 :=
by 
  sorry

end number_of_distinct_values_l38_38955


namespace total_ants_correct_l38_38587

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l38_38587


namespace students_play_both_l38_38619

variable (students total_students football cricket neither : ℕ)
variable (H1 : total_students = 420)
variable (H2 : football = 325)
variable (H3 : cricket = 175)
variable (H4 : neither = 50)
  
theorem students_play_both (H1 : total_students = 420) (H2 : football = 325) 
    (H3 : cricket = 175) (H4 : neither = 50) : 
    students = 325 + 175 - (420 - 50) :=
by sorry

end students_play_both_l38_38619


namespace determine_a_b_l38_38396

-- Step d) The Lean 4 statement for the transformed problem
theorem determine_a_b (a b : ℝ) (h : ∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2 * (a + t)^2 * 1 + t^2 + 3 * a * t + b = 0) : 
  a = 1 ∧ b = 1 := 
sorry

end determine_a_b_l38_38396


namespace abs_sum_less_than_two_l38_38575

theorem abs_sum_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) : |a + b| + |a - b| < 2 := 
sorry

end abs_sum_less_than_two_l38_38575


namespace maria_savings_after_purchase_l38_38142

theorem maria_savings_after_purchase
  (cost_sweater : ℕ)
  (cost_scarf : ℕ)
  (cost_mittens : ℕ)
  (num_family_members : ℕ)
  (savings : ℕ)
  (total_cost_one_set : ℕ)
  (total_cost_all_sets : ℕ)
  (amount_left : ℕ)
  (h1 : cost_sweater = 35)
  (h2 : cost_scarf = 25)
  (h3 : cost_mittens = 15)
  (h4 : num_family_members = 10)
  (h5 : savings = 800)
  (h6 : total_cost_one_set = cost_sweater + cost_scarf + cost_mittens)
  (h7 : total_cost_all_sets = total_cost_one_set * num_family_members)
  (h8 : amount_left = savings - total_cost_all_sets)
  : amount_left = 50 :=
sorry

end maria_savings_after_purchase_l38_38142


namespace positive_expression_l38_38291

theorem positive_expression (x y : ℝ) : (x^2 - 4 * x + y^2 + 13) > 0 := by
  sorry

end positive_expression_l38_38291


namespace inequality_proof_l38_38520

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (1 / Real.sqrt (x + y)) + (1 / Real.sqrt (y + z)) + (1 / Real.sqrt (z + x)) ≤ 1 / Real.sqrt (2 * x * y * z) :=
by
  sorry

end inequality_proof_l38_38520


namespace plane_intersects_unit_cubes_l38_38415

def unitCubeCount (side_length : ℕ) : ℕ :=
  side_length ^ 3

def intersectionCount (num_unitCubes : ℕ) (side_length : ℕ) : ℕ :=
  if side_length = 4 then 32 else 0 -- intersection count only applies for side_length = 4

theorem plane_intersects_unit_cubes
  (side_length : ℕ)
  (num_unitCubes : ℕ)
  (cubeArrangement : num_unitCubes = unitCubeCount side_length)
  (planeCondition : True) -- the plane is perpendicular to the diagonal and bisects it
  : intersectionCount num_unitCubes side_length = 32 := by
  sorry

end plane_intersects_unit_cubes_l38_38415


namespace necessary_not_sufficient_condition_l38_38134

-- Definitions of conditions
variable (x : ℝ)

-- Statement of the problem in Lean 4
theorem necessary_not_sufficient_condition (h : |x - 1| ≤ 1) : 2 - x ≥ 0 := sorry

end necessary_not_sufficient_condition_l38_38134


namespace cube_root_product_l38_38730

theorem cube_root_product : (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := 
by
  sorry

end cube_root_product_l38_38730


namespace cupcakes_left_l38_38153

theorem cupcakes_left (initial_cupcakes : ℕ)
  (students_delmont : ℕ) (ms_delmont : ℕ)
  (students_donnelly : ℕ) (mrs_donnelly : ℕ)
  (school_nurse : ℕ) (school_principal : ℕ) (school_custodians : ℕ)
  (favorite_teachers : ℕ) (cupcakes_per_favorite_teacher : ℕ)
  (other_classmates : ℕ) :
  initial_cupcakes = 80 →
  students_delmont = 18 → ms_delmont = 1 →
  students_donnelly = 16 → mrs_donnelly = 1 →
  school_nurse = 1 → school_principal = 1 → school_custodians = 3 →
  favorite_teachers = 5 → cupcakes_per_favorite_teacher = 2 → 
  other_classmates = 10 →
  initial_cupcakes - (students_delmont + ms_delmont +
                      students_donnelly + mrs_donnelly +
                      school_nurse + school_principal + school_custodians +
                      favorite_teachers * cupcakes_per_favorite_teacher +
                      other_classmates) = 19 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end cupcakes_left_l38_38153


namespace algebra_ineq_example_l38_38834

theorem algebra_ineq_example (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ( ( (xy + yz + zx) / 3 ) ^ (1/3) ) :=
by
  sorry

end algebra_ineq_example_l38_38834


namespace identity_eq_l38_38683

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l38_38683


namespace sufficient_condition_for_p_l38_38667

theorem sufficient_condition_for_p (m : ℝ) (h : 1 < m) : ∀ x : ℝ, x^2 - 2 * x + m > 0 :=
sorry

end sufficient_condition_for_p_l38_38667


namespace marcus_calzones_total_time_l38_38139

/-
Conditions:
1. It takes Marcus 20 minutes to saute the onions.
2. It takes a quarter of the time to saute the garlic and peppers that it takes to saute the onions.
3. It takes 30 minutes to knead the dough.
4. It takes twice as long to let the dough rest as it takes to knead it.
5. It takes 1/10th of the combined kneading and resting time to assemble the calzones.
-/

def time_saute_onions : ℕ := 20
def time_saute_garlic_peppers : ℕ := time_saute_onions / 4
def time_knead : ℕ := 30
def time_rest : ℕ := 2 * time_knead
def time_assemble : ℕ := (time_knead + time_rest) / 10

def total_time_making_calzones : ℕ :=
  time_saute_onions + time_saute_garlic_peppers + time_knead + time_rest + time_assemble

theorem marcus_calzones_total_time : total_time_making_calzones = 124 := by
  -- All steps and proof details to be filled in
  sorry

end marcus_calzones_total_time_l38_38139


namespace final_price_of_hat_is_correct_l38_38190

-- Definitions capturing the conditions.
def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

-- Calculations for the intermediate prices.
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

-- The theorem we need to prove.
theorem final_price_of_hat_is_correct : final_price = 9 := by
  sorry

end final_price_of_hat_is_correct_l38_38190


namespace other_root_of_quadratic_l38_38241

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) (h_root : 4 * a * 0^2 - 2 * a * 0 + c = 0) :
  ∃ t : ℝ, (4 * a * t^2 - 2 * a * t + c = 0) ∧ t = 1 / 2 :=
by
  sorry

end other_root_of_quadratic_l38_38241


namespace brittany_age_after_vacation_l38_38886

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end brittany_age_after_vacation_l38_38886


namespace a_8_value_l38_38473

variable {n : ℕ}
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_8_value : a 8 = 15 := by
  sorry

end a_8_value_l38_38473


namespace find_directrix_l38_38538

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := x^2 = 8 * y

-- State the problem to find the directrix of the given parabola
theorem find_directrix (x y : ℝ) (h : parabola_eq x y) : y = -2 :=
sorry

end find_directrix_l38_38538


namespace problem_statement_l38_38353

noncomputable def sequence_def (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  (a ≠ 0) ∧
  (S 1 = a) ∧
  (S 2 = 2 / S 1) ∧
  (∀ n, n ≥ 3 → S n = 2 / S (n - 1))

theorem problem_statement (a : ℝ) (S : ℕ → ℝ) (h : sequence_def a S 2018) : 
  S 2018 = 2 / a := 
by 
  sorry

end problem_statement_l38_38353


namespace proof_m_div_x_plus_y_l38_38141

variables (a b c x y m : ℝ)

-- 1. The ratio of 'a' to 'b' is 4 to 5
axiom h1 : a / b = 4 / 5

-- 2. 'c' is half of 'a'.
axiom h2 : c = a / 2

-- 3. 'x' equals 'a' increased by 27 percent of 'a'.
axiom h3 : x = 1.27 * a

-- 4. 'y' equals 'b' decreased by 16 percent of 'b'.
axiom h4 : y = 0.84 * b

-- 5. 'm' equals 'c' increased by 14 percent of 'c'.
axiom h5 : m = 1.14 * c

theorem proof_m_div_x_plus_y : m / (x + y) = 0.2457 :=
by
  -- Proof goes here
  sorry

end proof_m_div_x_plus_y_l38_38141


namespace sin_beta_value_l38_38420

open Real

theorem sin_beta_value (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h1 : sin α = 5 / 13) 
  (h2 : cos (α + β) = -4 / 5) : 
  sin β = 56 / 65 := 
sorry

end sin_beta_value_l38_38420


namespace height_of_water_in_cylinder_l38_38881

theorem height_of_water_in_cylinder
  (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V_cone : ℝ) (V_cylinder : ℝ) (h_cylinder : ℝ) :
  r_cone = 15 → h_cone = 25 → r_cylinder = 20 →
  V_cone = (1 / 3) * π * r_cone^2 * h_cone →
  V_cylinder = V_cone → V_cylinder = π * r_cylinder^2 * h_cylinder →
  h_cylinder = 4.7 :=
by
  intros r_cone_eq h_cone_eq r_cylinder_eq V_cone_eq V_cylinder_eq volume_eq
  sorry

end height_of_water_in_cylinder_l38_38881


namespace boxes_to_fill_l38_38887

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) (h₁ : total_boxes = 25.75) (h₂ : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := 
by
  sorry

end boxes_to_fill_l38_38887


namespace sum_of_squares_l38_38595

theorem sum_of_squares (a b c : ℝ) (h1 : ab + bc + ca = 4) (h2 : a + b + c = 17) : a^2 + b^2 + c^2 = 281 :=
by
  sorry

end sum_of_squares_l38_38595


namespace circles_intersect_l38_38238

-- Define the parameters and conditions given in the problem.
def r1 : ℝ := 5  -- Radius of circle O1
def r2 : ℝ := 8  -- Radius of circle O2
def d : ℝ := 8   -- Distance between the centers of O1 and O2

-- The main theorem that needs to be proven.
theorem circles_intersect (r1 r2 d : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 8) (h_d : d = 8) :
  r2 - r1 < d ∧ d < r1 + r2 :=
by
  sorry

end circles_intersect_l38_38238


namespace find_certain_number_l38_38217

theorem find_certain_number (x : ℝ) 
    (h : 7 * x - 6 - 12 = 4 * x) : x = 6 := 
by
  sorry

end find_certain_number_l38_38217


namespace largest_root_l38_38648

theorem largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -6) (h3 : p * q * r = -8) :
  max (max p q) r = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end largest_root_l38_38648


namespace cylindrical_to_rectangular_l38_38928

noncomputable def convertToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let r' := 2 * r
  let z' := z + 1
  convertToRectangular r' θ z' = (10, 10 * Real.sqrt 3, 3) :=
by
  sorry

end cylindrical_to_rectangular_l38_38928


namespace weight_of_b_l38_38895

variable (Wa Wb Wc: ℝ)

-- Conditions
def avg_weight_abc : Prop := (Wa + Wb + Wc) / 3 = 45
def avg_weight_ab : Prop := (Wa + Wb) / 2 = 40
def avg_weight_bc : Prop := (Wb + Wc) / 2 = 43

-- Theorem to prove
theorem weight_of_b (Wa Wb Wc: ℝ) (h_avg_abc : avg_weight_abc Wa Wb Wc)
  (h_avg_ab : avg_weight_ab Wa Wb) (h_avg_bc : avg_weight_bc Wb Wc) : Wb = 31 :=
by
  sorry

end weight_of_b_l38_38895


namespace keesha_total_cost_is_correct_l38_38355

noncomputable def hair_cost : ℝ := 
  let cost := 50.0 
  let discount := cost * 0.10 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def nails_cost : ℝ := 
  let manicure_cost := 30.0 
  let pedicure_cost := 35.0 * 0.50 
  let total_without_tip := manicure_cost + pedicure_cost 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def makeup_cost : ℝ := 
  let cost := 40.0 
  let tax := cost * 0.07 
  let total_without_tip := cost + tax 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def facial_cost : ℝ := 
  let cost := 60.0 
  let discount := cost * 0.15 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def total_cost : ℝ := 
  hair_cost + nails_cost + makeup_cost + facial_cost

theorem keesha_total_cost_is_correct : total_cost = 223.56 := by
  sorry

end keesha_total_cost_is_correct_l38_38355


namespace problem1_arithmetic_sequence_problem2_geometric_sequence_l38_38409

-- Problem (1)
variable (S : Nat → Int)
variable (a : Nat → Int)

axiom S10_eq_50 : S 10 = 50
axiom S20_eq_300 : S 20 = 300
axiom S_def : (∀ n : Nat, n > 0 → S n = n * a 1 + (n * (n-1) / 2) * (a 2 - a 1))

theorem problem1_arithmetic_sequence (n : Nat) : a n = 2 * n - 6 := sorry

-- Problem (2)
variable (a : Nat → Int)

axiom S3_eq_a2_plus_10a1 : S 3 = a 2 + 10 * a 1
axiom a5_eq_81 : a 5 = 81
axiom positive_terms : ∀ n, a n > 0

theorem problem2_geometric_sequence (n : Nat) : S n = (3 ^ n - 1) / 2 := sorry

end problem1_arithmetic_sequence_problem2_geometric_sequence_l38_38409


namespace repayment_is_correct_l38_38902

noncomputable def repayment_amount (a r : ℝ) : ℝ := a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1)

theorem repayment_is_correct (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  repayment_amount a r = a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1) :=
by
  sorry

end repayment_is_correct_l38_38902


namespace ellipse_distance_CD_l38_38943

theorem ellipse_distance_CD :
  ∃ (CD : ℝ), 
    (∀ (x y : ℝ),
    4 * (x - 2)^2 + 16 * y^2 = 64) → 
      CD = 2*Real.sqrt 5 :=
by sorry

end ellipse_distance_CD_l38_38943


namespace jezebel_total_flower_cost_l38_38758

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end jezebel_total_flower_cost_l38_38758


namespace binom_16_12_eq_1820_l38_38213

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l38_38213


namespace smallest_value_of_y_l38_38627

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end smallest_value_of_y_l38_38627


namespace max_value_of_symmetric_f_l38_38249

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l38_38249


namespace find_k_value_l38_38885

theorem find_k_value :
  ∃ k : ℝ, (∀ x : ℝ, 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5) ∧ 
          (∃ a b : ℝ, b - a = 8 ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5)) ∧ 
          k = 9 / 4 :=
sorry

end find_k_value_l38_38885


namespace lily_pad_half_lake_l38_38968

theorem lily_pad_half_lake
  (P : ℕ → ℝ) -- Define a function P(n) which represents the size of the patch on day n.
  (h1 : ∀ n, P n = P (n - 1) * 2) -- Every day, the patch doubles in size.
  (h2 : P 58 = 1) -- It takes 58 days for the patch to cover the entire lake (normalized to 1).
  : P 57 = 1 / 2 :=
by
  sorry

end lily_pad_half_lake_l38_38968


namespace system1_solution_system2_solution_l38_38171

-- System (1)
theorem system1_solution {x y : ℝ} : 
  x + y = 3 → 
  x - y = 1 → 
  (x = 2 ∧ y = 1) :=
by
  intros h1 h2
  -- proof goes here
  sorry

-- System (2)
theorem system2_solution {x y : ℝ} :
  2 * x + y = 3 →
  x - 2 * y = 1 →
  (x = 7 / 5 ∧ y = 1 / 5) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end system1_solution_system2_solution_l38_38171


namespace distance_between_points_l38_38684

theorem distance_between_points (x : ℝ) :
  let M := (-1, 4)
  let N := (x, 4)
  dist (M, N) = 5 →
  (x = -6 ∨ x = 4) := sorry

end distance_between_points_l38_38684


namespace find_x_l38_38607

theorem find_x (x m n : ℤ) 
  (h₁ : 15 + x = m^2) 
  (h₂ : x - 74 = n^2) :
  x = 2010 :=
by
  sorry

end find_x_l38_38607


namespace y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l38_38345

-- Definitions of cost functions for travel agencies
def full_ticket_price : ℕ := 240

def y_A (x : ℕ) : ℕ := 120 * x + 240
def y_B (x : ℕ) : ℕ := 144 * x + 144

-- Prove functional relationships for y_A and y_B
theorem y_A_functional_relationship (x : ℕ) : y_A x = 120 * x + 240 :=
by sorry

theorem y_B_functional_relationship (x : ℕ) : y_B x = 144 * x + 144 :=
by sorry

-- Prove conditions for cost-effectiveness
theorem cost_effective_B (x : ℕ) : x < 4 → y_A x > y_B x :=
by sorry

theorem cost_effective_equal (x : ℕ) : x = 4 → y_A x = y_B x :=
by sorry

theorem cost_effective_A (x : ℕ) : x > 4 → y_A x < y_B x :=
by sorry

end y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l38_38345


namespace find_pairs_eq_l38_38742

theorem find_pairs_eq : 
  { (m, n) : ℕ × ℕ | 0 < m ∧ 0 < n ∧ m ^ 2 + 2 * n ^ 2 = 3 * (m + 2 * n) } = {(3, 3), (4, 2)} :=
by sorry

end find_pairs_eq_l38_38742


namespace value_of_m_l38_38882

theorem value_of_m 
  (m : ℤ) 
  (h : ∀ x : ℤ, x^2 - 2 * (m + 1) * x + 16 = (x - 4)^2) : 
  m = 3 := 
sorry

end value_of_m_l38_38882


namespace boxes_left_l38_38350

-- Define the initial number of boxes
def initial_boxes : ℕ := 10

-- Define the number of boxes sold
def boxes_sold : ℕ := 5

-- Define a theorem stating that the number of boxes left is 5
theorem boxes_left : initial_boxes - boxes_sold = 5 :=
by
  sorry

end boxes_left_l38_38350


namespace copper_percentage_l38_38357

theorem copper_percentage (copperFirst copperSecond totalWeight1 totalWeight2: ℝ) 
    (h1 : copperFirst = 0.25)
    (h2 : copperSecond = 0.50) 
    (h3 : totalWeight1 = 200) 
    (h4 : totalWeight2 = 800) : 
    (copperFirst * totalWeight1 + copperSecond * totalWeight2) / (totalWeight1 + totalWeight2) * 100 = 45 := 
by 
  sorry

end copper_percentage_l38_38357


namespace relationship_between_a_b_l38_38892

theorem relationship_between_a_b (a b c : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -2)
  (h3 : a * x + c * y = 1) (h4 : c * x - b * y = 2) : 9 * a + 4 * b = 1 :=
sorry

end relationship_between_a_b_l38_38892


namespace probability_A_seventh_week_l38_38210

/-
Conditions:
1. There are four different passwords: A, B, C, and D.
2. Each week, one of these passwords is used.
3. Each week, the password is chosen at random and equally likely from the three passwords that were not used in the previous week.
4. Password A is used in the first week.

Goal:
Prove that the probability that password A will be used in the seventh week is 61/243.
-/

def prob_password_A_in_seventh_week : ℚ :=
  let Pk (k : ℕ) : ℚ := 
    if k = 1 then 1
    else if k >= 2 then ((-1 / 3)^(k - 1) * (3 / 4) + 1 / 4) else 0
  Pk 7

theorem probability_A_seventh_week : prob_password_A_in_seventh_week = 61 / 243 := by
  sorry

end probability_A_seventh_week_l38_38210


namespace log_comparison_l38_38135

noncomputable def logBase (a x : ℝ) := Real.log x / Real.log a

theorem log_comparison
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) 
  (m : ℝ) (hm : m = logBase a (a^2 + 1))
  (n : ℝ) (hn : n = logBase a (a + 1))
  (p : ℝ) (hp : p = logBase a (2 * a)) :
  p > m ∧ m > n :=
by
  sorry

end log_comparison_l38_38135


namespace determine_ordered_pair_l38_38727

theorem determine_ordered_pair (s n : ℤ)
    (h1 : ∀ t : ℤ, ∃ x y : ℤ,
        (x, y) = (s + 2 * t, -3 + n * t)) 
    (h2 : ∀ x y : ℤ, y = 2 * x - 7) :
    (s, n) = (2, 4) :=
by
  sorry

end determine_ordered_pair_l38_38727


namespace cost_of_article_l38_38850

noncomputable def find_cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : Prop :=
  C = 168.57

theorem cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : 
  find_cost_of_article C G h1 h2 :=
by
  sorry

end cost_of_article_l38_38850


namespace trapezoid_side_lengths_l38_38095

theorem trapezoid_side_lengths
  (isosceles : ∀ (A B C D : ℝ) (height BE : ℝ), height = 2 → BE = 2 → A = 2 * Real.sqrt 2 → D = A → 12 = 0.5 * (B + C) * BE → A = D)
  (area : ∀ (BC AD : ℝ), 12 = 0.5 * (BC + AD) * 2)
  (height : ∀ (BE : ℝ), BE = 2)
  (intersect_right_angle : ∀ (A B C D : ℝ), 90 = 45 + 45) :
  ∃ A B C D, A = 2 * Real.sqrt 2 ∧ B = 4 ∧ C = 8 ∧ D = 2 * Real.sqrt 2 :=
by
  sorry

end trapezoid_side_lengths_l38_38095


namespace cost_price_equals_selling_price_l38_38889

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (h1 : 20 * C = 1.25 * C * x) : x = 16 :=
by
  -- This proof is omitted at the moment
  sorry

end cost_price_equals_selling_price_l38_38889


namespace train_speed_l38_38578

theorem train_speed (L1 L2 : ℕ) (V2 : ℕ) (t : ℝ) (V1 : ℝ) : 
  L1 = 200 → 
  L2 = 280 → 
  V2 = 30 → 
  t = 23.998 → 
  (0.001 * (L1 + L2)) / (t / 3600) = V1 + V2 → 
  V1 = 42 :=
by 
  intros
  sorry

end train_speed_l38_38578


namespace solution_of_equation_l38_38762

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solution_of_equation (k : ℤ) (h : -1 ≤ k ∧ k ≤ 5) :
  ∃ x : ℝ, 4 * ↑(integer_part x) = 25 * fractional_part x - 4.5 ∧
           x = k + (8 * ↑k + 9) / 50 := 
sorry

end solution_of_equation_l38_38762


namespace hotdogs_sold_correct_l38_38070

def initial_hotdogs : ℕ := 99
def remaining_hotdogs : ℕ := 97
def sold_hotdogs : ℕ := initial_hotdogs - remaining_hotdogs

theorem hotdogs_sold_correct : sold_hotdogs = 2 := by
  sorry

end hotdogs_sold_correct_l38_38070


namespace minimum_value_of_C_over_D_is_three_l38_38097

variable (x : ℝ) (C D : ℝ)
variables (hxC : x^3 + 1/(x^3) = C) (hxD : x - 1/(x) = D)

theorem minimum_value_of_C_over_D_is_three (hC : C = D^3 + 3 * D) :
  ∃ x : ℝ, x^3 + 1/(x^3) = C ∧ x - 1/(x) = D → C / D ≥ 3 :=
by
  sorry

end minimum_value_of_C_over_D_is_three_l38_38097


namespace undefined_values_l38_38552

theorem undefined_values (b : ℝ) : (b^2 - 9 = 0) ↔ (b = -3 ∨ b = 3) := by
  sorry

end undefined_values_l38_38552


namespace ammonium_iodide_required_l38_38470

theorem ammonium_iodide_required
  (KOH_moles NH3_moles KI_moles H2O_moles : ℕ)
  (hn : NH3_moles = 3) (hk : KOH_moles = 3) (hi : KI_moles = 3) (hw : H2O_moles = 3) :
  ∃ NH4I_moles, NH3_moles = 3 ∧ KI_moles = 3 ∧ H2O_moles = 3 ∧ KOH_moles = 3 ∧ NH4I_moles = 3 :=
by
  sorry

end ammonium_iodide_required_l38_38470


namespace initial_concentration_of_hydrochloric_acid_l38_38084

theorem initial_concentration_of_hydrochloric_acid
  (initial_mass : ℕ)
  (drained_mass : ℕ)
  (added_concentration : ℕ)
  (final_concentration : ℕ)
  (total_mass : ℕ)
  (initial_concentration : ℕ) :
  initial_mass = 300 ∧ drained_mass = 25 ∧ added_concentration = 80 ∧ final_concentration = 25 ∧ total_mass = 300 →
  (275 * initial_concentration / 100 + 20 = 75) →
  initial_concentration = 20 :=
by
  intros h_eq h_new_solution
  -- Rewriting the data given in h_eq and solving h_new_solution
  rcases h_eq with ⟨h_initial_mass, h_drained_mass, h_added_concentration, h_final_concentration, h_total_mass⟩
  sorry

end initial_concentration_of_hydrochloric_acid_l38_38084


namespace total_players_correct_l38_38788

-- Define the number of players for each type of sport
def cricket_players : Nat := 12
def hockey_players : Nat := 17
def football_players : Nat := 11
def softball_players : Nat := 10

-- The theorem we aim to prove
theorem total_players_correct : 
  cricket_players + hockey_players + football_players + softball_players = 50 := by
  sorry

end total_players_correct_l38_38788


namespace correct_statements_about_microbial_counting_l38_38760

def hemocytometer_counts_bacteria_or_yeast : Prop :=
  true -- based on condition 1

def plate_streaking_allows_colony_counting : Prop :=
  false -- count is not done using the plate streaking method, based on the analysis

def dilution_plating_allows_colony_counting : Prop :=
  true -- based on condition 3  
  
def dilution_plating_count_is_accurate : Prop :=
  false -- colony count is often lower than the actual number, based on the analysis

theorem correct_statements_about_microbial_counting :
  (hemocytometer_counts_bacteria_or_yeast ∧ dilution_plating_allows_colony_counting)
= (plate_streaking_allows_colony_counting ∨ dilution_plating_count_is_accurate) :=
by sorry

end correct_statements_about_microbial_counting_l38_38760


namespace carl_max_rocks_value_l38_38184

/-- 
Carl finds rocks of three different types:
  - 6-pound rocks worth $18 each.
  - 3-pound rocks worth $9 each.
  - 2-pound rocks worth $3 each.
There are at least 15 rocks available for each type.
Carl can carry at most 20 pounds.

Prove that the maximum value, in dollars, of the rocks Carl can carry out of the cave is $57.
-/
theorem carl_max_rocks_value : 
  (∃ x y z : ℕ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 6 * x + 3 * y + 2 * z ≤ 20 ∧ 18 * x + 9 * y + 3 * z = 57) :=
sorry

end carl_max_rocks_value_l38_38184


namespace smallest_positive_solution_to_congruence_l38_38842

theorem smallest_positive_solution_to_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 33] ∧ x = 28 := 
by 
  sorry

end smallest_positive_solution_to_congruence_l38_38842


namespace more_visitors_that_day_l38_38847

def number_of_visitors_previous_day : ℕ := 100
def number_of_visitors_that_day : ℕ := 666

theorem more_visitors_that_day :
  number_of_visitors_that_day - number_of_visitors_previous_day = 566 :=
by
  sorry

end more_visitors_that_day_l38_38847


namespace infinite_primes_dividing_S_l38_38397

noncomputable def infinite_set_of_pos_integers (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S) ∧ ∀ n : ℕ, n ∈ S → n > 0

def set_of_sums (S : Set ℕ) : Set ℕ :=
  {t | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ t = x + y}

noncomputable def finitely_many_primes_condition (S : Set ℕ) (T : Set ℕ) : Prop :=
  {p : ℕ | Prime p ∧ p % 4 = 1 ∧ (∃ t ∈ T, p ∣ t)}.Finite

theorem infinite_primes_dividing_S (S : Set ℕ) (T := set_of_sums S)
  (hS : infinite_set_of_pos_integers S)
  (hT : finitely_many_primes_condition S T) :
  {p : ℕ | Prime p ∧ ∃ s ∈ S, p ∣ s}.Infinite := 
sorry

end infinite_primes_dividing_S_l38_38397


namespace probability_diff_colors_l38_38000

theorem probability_diff_colors (total_balls red_balls white_balls selected_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_red : red_balls = 2)
  (h_white : white_balls = 2)
  (h_selected : selected_balls = 2) :
  (∃ P : ℚ, P = (red_balls.choose (selected_balls / 2) * white_balls.choose (selected_balls / 2)) / total_balls.choose selected_balls ∧ P = 2 / 3) :=
by 
  sorry

end probability_diff_colors_l38_38000


namespace sin_sum_given_cos_tan_conditions_l38_38508

open Real

theorem sin_sum_given_cos_tan_conditions 
  (α β : ℝ)
  (h1 : cos α + cos β = 1 / 3)
  (h2 : tan (α + β) = 24 / 7)
  : sin α + sin β = 1 / 4 ∨ sin α + sin β = -4 / 9 := 
  sorry

end sin_sum_given_cos_tan_conditions_l38_38508


namespace arithmetic_sequence_lemma_l38_38583

theorem arithmetic_sequence_lemma (a : ℕ → ℝ) (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0)
  (h_condition : a 3 + a 11 = 22) : a 7 = 11 :=
sorry

end arithmetic_sequence_lemma_l38_38583


namespace find_k_value_l38_38864

noncomputable def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| n => a + (n - 1) * d

theorem find_k_value (a d : ℤ) (k : ℕ) 
  (h1 : arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 24)
  (h2 : (Finset.range 11).sum (λ i => arithmetic_seq a d (5 + i)) = 110)
  (h3 : arithmetic_seq a d k = 16) : 
  k = 16 :=
sorry

end find_k_value_l38_38864


namespace ratio_area_of_rectangle_to_square_l38_38496

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l38_38496


namespace hari_contribution_l38_38474

theorem hari_contribution 
    (P_investment : ℕ) (P_time : ℕ) (H_time : ℕ) (profit_ratio : ℚ)
    (investment_ratio : P_investment * P_time / (Hari_contribution * H_time) = profit_ratio) :
    Hari_contribution = 10080 :=
by
    have P_investment := 3920
    have P_time := 12
    have H_time := 7
    have profit_ratio := (2 : ℚ) / 3
    sorry

end hari_contribution_l38_38474


namespace first_cube_weight_l38_38200

-- Given definitions of cubes and their relationships
def weight_of_cube (s : ℝ) (weight : ℝ) : Prop :=
  ∃ v : ℝ, v = s^3 ∧ weight = v

def cube_relationship (s1 s2 weight2 : ℝ) : Prop :=
  s2 = 2 * s1 ∧ weight2 = 32

-- The proof problem
theorem first_cube_weight (s1 s2 weight1 weight2 : ℝ) (h1 : cube_relationship s1 s2 weight2) : weight1 = 4 :=
  sorry

end first_cube_weight_l38_38200


namespace product_in_base_7_l38_38838

def base_7_product : ℕ :=
  let b := 7
  Nat.ofDigits b [3, 5, 6] * Nat.ofDigits b [4]

theorem product_in_base_7 :
  base_7_product = Nat.ofDigits 7 [3, 2, 3, 1, 2] :=
by
  -- The proof is formally skipped for this exercise, hence we insert 'sorry'.
  sorry

end product_in_base_7_l38_38838


namespace cary_initial_wage_l38_38483

noncomputable def initial_hourly_wage (x : ℝ) : Prop :=
  let first_year_wage := 1.20 * x
  let second_year_wage := 0.75 * first_year_wage
  second_year_wage = 9

theorem cary_initial_wage : ∃ x : ℝ, initial_hourly_wage x ∧ x = 10 := 
by
  use 10
  unfold initial_hourly_wage
  simp
  sorry

end cary_initial_wage_l38_38483


namespace min_CD_squared_diff_l38_38844

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 2)) + (Real.sqrt (z + 2))
noncomputable def f (x y z : ℝ) : ℝ := (C x y z) ^ 2 - (D x y z) ^ 2

theorem min_CD_squared_diff (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  f x y z ≥ 41.4736 :=
sorry

end min_CD_squared_diff_l38_38844


namespace other_root_of_quadratic_l38_38919

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l38_38919


namespace triangle_areas_l38_38597

theorem triangle_areas (r s : ℝ) (h1 : s = (1/2) * r + 6)
                       (h2 : (12 + r) * ((1/2) * r + 6) = 18) :
  r + s = -3 :=
by
  sorry

end triangle_areas_l38_38597


namespace shaded_region_area_l38_38035

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end shaded_region_area_l38_38035


namespace initial_walking_rate_proof_l38_38275

noncomputable def initial_walking_rate (d : ℝ) (v_miss : ℝ) (t_miss : ℝ) (v_early : ℝ) (t_early : ℝ) : ℝ :=
  d / ((d / v_early) + t_early - t_miss)

theorem initial_walking_rate_proof :
  initial_walking_rate 6 5 (7/60) 6 (5/60) = 5 := by
  sorry

end initial_walking_rate_proof_l38_38275


namespace m_perp_n_α_perp_β_l38_38068

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Problem 1:
axiom m_perp_α : perpendicular_to_plane m α
axiom n_perp_β : perpendicular_to_plane n β
axiom α_perp_β : perpendicular_planes α β

theorem m_perp_n : perpendicular_lines m n :=
sorry

-- Problem 2:
axiom m_perp_n' : perpendicular_lines m n
axiom m_perp_α' : perpendicular_to_plane m α
axiom n_perp_β' : perpendicular_to_plane n β

theorem α_perp_β' : perpendicular_planes α β :=
sorry

end m_perp_n_α_perp_β_l38_38068


namespace speed_conversion_l38_38302

theorem speed_conversion (v : ℚ) (h : v = 9/36) : v * 3.6 = 0.9 := by
  sorry

end speed_conversion_l38_38302


namespace minimum_distance_from_mars_l38_38606

noncomputable def distance_function (a b c t : ℝ) : ℝ :=
  a * t^2 + b * t + c

theorem minimum_distance_from_mars :
  ∃ t₀ : ℝ, distance_function (11/54) (-1/18) 4 t₀ = (9:ℝ) :=
  sorry

end minimum_distance_from_mars_l38_38606


namespace find_multiple_of_larger_integer_l38_38245

/--
The sum of two integers is 30. A certain multiple of the larger integer is 10 less than 5 times
the smaller integer. The smaller integer is 10. What is the multiple of the larger integer?
-/
theorem find_multiple_of_larger_integer
  (S L M : ℤ)
  (h1 : S + L = 30)
  (h2 : S = 10)
  (h3 : M * L = 5 * S - 10) :
  M = 2 :=
sorry

end find_multiple_of_larger_integer_l38_38245


namespace other_denominations_l38_38565

theorem other_denominations :
  ∀ (total_checks : ℕ) (total_value : ℝ) (fifty_denomination_checks : ℕ) (remaining_avg : ℝ),
    total_checks = 30 →
    total_value = 1800 →
    fifty_denomination_checks = 15 →
    remaining_avg = 70 →
    ∃ (other_denomination : ℝ), other_denomination = 70 :=
by
  intros total_checks total_value fifty_denomination_checks remaining_avg
  intros h1 h2 h3 h4
  let other_denomination := 70
  use other_denomination
  sorry

end other_denominations_l38_38565


namespace total_money_spent_l38_38765

noncomputable def total_expenditure (A : ℝ) : ℝ :=
  let person1_8_expenditure := 8 * 12
  let person9_expenditure := A + 8
  person1_8_expenditure + person9_expenditure

theorem total_money_spent :
  (∃ A : ℝ, total_expenditure A = 9 * A ∧ A = 13) →
  total_expenditure 13 = 117 :=
by
  intro h
  sorry

end total_money_spent_l38_38765


namespace percentage_not_red_roses_l38_38598

-- Definitions for the conditions
def roses : Nat := 25
def tulips : Nat := 40
def daisies : Nat := 60
def lilies : Nat := 15
def sunflowers : Nat := 10
def totalFlowers : Nat := roses + tulips + daisies + lilies + sunflowers -- 150
def redRoses : Nat := roses / 2 -- 12 (considering integer division)

-- Statement to prove
theorem percentage_not_red_roses : 
  ((totalFlowers - redRoses) * 100 / totalFlowers) = 92 := by
  sorry

end percentage_not_red_roses_l38_38598


namespace johns_salary_percentage_increase_l38_38995

theorem johns_salary_percentage_increase (initial_salary final_salary : ℕ) (h1 : initial_salary = 50) (h2 : final_salary = 90) :
  ((final_salary - initial_salary : ℕ) / initial_salary : ℚ) * 100 = 80 := by
  sorry

end johns_salary_percentage_increase_l38_38995


namespace catch_up_time_l38_38326

theorem catch_up_time (x : ℕ) : 240 * x = 150 * x + 12 * 150 := by
  sorry

end catch_up_time_l38_38326


namespace fraction_of_work_left_l38_38572

theorem fraction_of_work_left 
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ) 
  (A_rate : ℚ := 1 / A_days) (B_rate : ℚ := 1 / B_days) (combined_rate : ℚ := 1 / A_days + 1 / B_days) 
  (work_completed : ℚ := combined_rate * work_days) (fraction_left : ℚ := 1 - work_completed)
  (hA : A_days = 15) (hB : B_days = 20) (hW : work_days = 4) 
  : fraction_left = 8 / 15 :=
sorry

end fraction_of_work_left_l38_38572


namespace female_students_selected_l38_38201

theorem female_students_selected (males females : ℕ) (p : ℚ) (h_males : males = 28)
  (h_females : females = 21) (h_p : p = 1 / 7) : females * p = 3 := by 
  sorry

end female_students_selected_l38_38201


namespace consecutive_even_numbers_average_35_greatest_39_l38_38087

-- Defining the conditions of the problem
def average_of_even_numbers (n : ℕ) (S : ℕ) : ℕ := (n * S + (2 * n * (n - 1)) / 2) / n

-- Main statement to be proven
theorem consecutive_even_numbers_average_35_greatest_39 : 
  ∃ (n : ℕ), average_of_even_numbers n (38 - (n - 1) * 2) = 35 ∧ (38 - (n - 1) * 2) + (n - 1) * 2 = 38 :=
by
  sorry

end consecutive_even_numbers_average_35_greatest_39_l38_38087


namespace sector_radius_l38_38128

theorem sector_radius (l : ℝ) (a : ℝ) (r : ℝ) (h1 : l = 2) (h2 : a = 4) (h3 : a = (1 / 2) * l * r) : r = 4 := by
  sorry

end sector_radius_l38_38128


namespace minimize_x_expr_minimized_l38_38257

noncomputable def minimize_x_expr (x : ℝ) : ℝ :=
  x + 4 / (x + 1)

theorem minimize_x_expr_minimized 
  (hx : x > -1) 
  : x = 1 ↔ minimize_x_expr x = minimize_x_expr 1 :=
by
  sorry

end minimize_x_expr_minimized_l38_38257


namespace abigail_writing_time_l38_38735

def total_additional_time (words_needed : ℕ) (words_per_half_hour : ℕ) (words_already_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let remaining_words := words_needed - words_already_written
  let half_hour_blocks := (remaining_words + words_per_half_hour - 1) / words_per_half_hour -- ceil(remaining_words / words_per_half_hour)
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

theorem abigail_writing_time :
  total_additional_time 1500 250 200 45 = 225 :=
by {
  -- Adding the proof in Lean:
  -- fail to show you the detailed steps, hence added sorry
  sorry
}

end abigail_writing_time_l38_38735


namespace exercise_l38_38824

theorem exercise (x y z : ℕ) (h1 : x * y * z = 1) : (7 ^ ((x + y + z) ^ 3) / 7 ^ ((x - y + z) ^ 3)) = 7 ^ 6 := 
by
  sorry

end exercise_l38_38824


namespace sum_of_four_primes_is_prime_l38_38585

theorem sum_of_four_primes_is_prime
    (A B : ℕ)
    (hA_prime : Prime A)
    (hB_prime : Prime B)
    (hA_minus_B_prime : Prime (A - B))
    (hA_plus_B_prime : Prime (A + B)) :
    Prime (A + B + (A - B) + A) :=
by
  sorry

end sum_of_four_primes_is_prime_l38_38585


namespace cos_alpha_plus_pi_six_l38_38511

theorem cos_alpha_plus_pi_six (α : ℝ) (hα_in_interval : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 3 / 3) :
  Real.cos (α + π / 6) = (3 - Real.sqrt 6) / 6 := 
by
  sorry

end cos_alpha_plus_pi_six_l38_38511


namespace find_initial_girls_l38_38610

variable (b g : ℕ)

theorem find_initial_girls 
  (h1 : 3 * (g - 18) = b)
  (h2 : 4 * (b - 36) = g - 18) :
  g = 31 := 
by
  sorry

end find_initial_girls_l38_38610


namespace babysitting_earnings_l38_38041

theorem babysitting_earnings
  (cost_video_game : ℕ)
  (cost_candy : ℕ)
  (hours_worked : ℕ)
  (amount_left : ℕ)
  (total_earned : ℕ)
  (earnings_per_hour : ℕ) :
  cost_video_game = 60 →
  cost_candy = 5 →
  hours_worked = 9 →
  amount_left = 7 →
  total_earned = cost_video_game + cost_candy + amount_left →
  earnings_per_hour = total_earned / hours_worked →
  earnings_per_hour = 8 :=
by
  intros h_game h_candy h_hours h_left h_total_earned h_earn_per_hour
  rw [h_game, h_candy] at h_total_earned
  simp at h_total_earned
  have h_total_earned : total_earned = 72 := by linarith
  rw [h_total_earned, h_hours] at h_earn_per_hour
  simp at h_earn_per_hour
  assumption

end babysitting_earnings_l38_38041


namespace slower_train_passing_time_l38_38752

/--
Two goods trains, each 500 meters long, are running in opposite directions on parallel tracks. 
Their respective speeds are 45 kilometers per hour and 15 kilometers per hour. 
Prove that the time taken by the slower train to pass the driver of the faster train is 30 seconds.
-/
theorem slower_train_passing_time : 
  ∀ (distance length_speed : ℝ), 
    distance = 500 →
    ∃ (v1 v2 : ℝ), 
      v1 = 45 * (1000 / 3600) → 
      v2 = 15 * (1000 / 3600) →
      (distance / ((v1 + v2) * (3/50)) = 30) :=
by
  sorry

end slower_train_passing_time_l38_38752


namespace end_same_digit_l38_38863

theorem end_same_digit
  (a b : ℕ)
  (h : (2 * a + b) % 10 = (2 * b + a) % 10) :
  a % 10 = b % 10 :=
by
  sorry

end end_same_digit_l38_38863


namespace shopkeeper_total_profit_percentage_l38_38923

noncomputable def profit_percentage (actual_weight faulty_weight ratio : ℕ) : ℝ :=
  (actual_weight - faulty_weight) / actual_weight * 100 * ratio

noncomputable def total_profit_percentage (ratios profits : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) ratios profits)) / (List.sum ratios)

theorem shopkeeper_total_profit_percentage :
  let actual_weight := 1000
  let faulty_weights := [900, 850, 950]
  let profit_percentages := [10, 15, 5]
  let ratios := [3, 2, 1]
  total_profit_percentage ratios profit_percentages = 10.83 :=
by
  sorry

end shopkeeper_total_profit_percentage_l38_38923


namespace part_a_area_of_square_l38_38209

theorem part_a_area_of_square {s : ℝ} (h : s = 9) : s ^ 2 = 81 := 
sorry

end part_a_area_of_square_l38_38209


namespace dampening_factor_l38_38390

theorem dampening_factor (s r : ℝ) 
  (h1 : s / (1 - r) = 16) 
  (h2 : s * r / (1 - r^2) = -6) :
  r = -3 / 11 := 
sorry

end dampening_factor_l38_38390


namespace rectangle_perimeter_is_3y_l38_38003

noncomputable def congruent_rectangle_perimeter (y : ℝ) (h1 : y > 0) : ℝ :=
  let side_length := 2 * y
  let center_square_side := y
  let width := (side_length - center_square_side) / 2
  let length := center_square_side
  2 * (length + width)

theorem rectangle_perimeter_is_3y (y : ℝ) (h1 : y > 0) :
  congruent_rectangle_perimeter y h1 = 3 * y :=
sorry

end rectangle_perimeter_is_3y_l38_38003


namespace valid_bases_for_625_l38_38094

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end valid_bases_for_625_l38_38094


namespace lucy_times_three_ago_l38_38091

  -- Defining the necessary variables and conditions
  def lucy_age_now : ℕ := 50
  def lovely_age (x : ℕ) : ℕ := 20  -- The age of Lovely when x years has passed
  
  -- Statement of the problem
  theorem lucy_times_three_ago {x : ℕ} : 
    (lucy_age_now - x = 3 * (lovely_age x - x)) → (lucy_age_now + 10 = 2 * (lovely_age x + 10)) → x = 5 := 
  by
  -- Proof is omitted
  sorry
  
end lucy_times_three_ago_l38_38091


namespace odd_lattice_points_on_BC_l38_38980

theorem odd_lattice_points_on_BC
  (A B C : ℤ × ℤ)
  (odd_lattice_points_AB : Odd ((B.1 - A.1) * (B.2 - A.2)))
  (odd_lattice_points_AC : Odd ((C.1 - A.1) * (C.2 - A.2))) :
  Odd ((C.1 - B.1) * (C.2 - B.2)) :=
sorry

end odd_lattice_points_on_BC_l38_38980


namespace conic_section_is_hyperbola_l38_38678

theorem conic_section_is_hyperbola : 
  ∀ (x y : ℝ), x^2 + 2 * x - 8 * y^2 = 0 → (∃ a b h k : ℝ, (x + 1)^2 / a^2 - (y - 0)^2 / b^2 = 1) := 
by 
  intros x y h_eq;
  sorry

end conic_section_is_hyperbola_l38_38678


namespace price_of_72_cans_l38_38706

def regular_price_per_can : ℝ := 0.30
def discount_percentage : ℝ := 0.15
def discounted_price_per_can := regular_price_per_can * (1 - discount_percentage)
def cans_purchased : ℕ := 72

theorem price_of_72_cans :
  cans_purchased * discounted_price_per_can = 18.36 :=
by sorry

end price_of_72_cans_l38_38706


namespace paige_finished_problems_l38_38030

-- Define the conditions
def initial_problems : ℕ := 110
def problems_per_page : ℕ := 9
def remaining_pages : ℕ := 7

-- Define the statement we want to prove
theorem paige_finished_problems :
  initial_problems - (remaining_pages * problems_per_page) = 47 :=
by sorry

end paige_finished_problems_l38_38030


namespace train_length_proof_l38_38364

noncomputable def train_length (speed_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

theorem train_length_proof : train_length 100 18 = 500.04 :=
  sorry

end train_length_proof_l38_38364


namespace snow_total_inches_l38_38832

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end snow_total_inches_l38_38832


namespace percentage_increase_equal_price_l38_38078

/-
A merchant has selected two items to be placed on sale, one of which currently sells for 20 percent less than the other.
He wishes to raise the price of the cheaper item so that the two items are equally priced.
By what percentage must he raise the price of the less expensive item?
-/
theorem percentage_increase_equal_price (P: ℝ) : (P > 0) → 
  (∀ cheap_item, cheap_item = 0.80 * P → ((P - cheap_item) / cheap_item) * 100 = 25) :=
by
  intro P_pos
  intro cheap_item
  intro h
  sorry

end percentage_increase_equal_price_l38_38078


namespace min_a_condition_l38_38096

-- Definitions of the conditions
def real_numbers (x : ℝ) := true

def in_interval (a m n : ℝ) : Prop := 0 < n ∧ n < m ∧ m < 1 / a

def inequality (a m n : ℝ) : Prop :=
  (n^(1/m) / m^(1/n) > (n^a) / (m^a))

-- Lean statement
theorem min_a_condition (a m n : ℝ) (h1 : real_numbers m) (h2 : real_numbers n)
    (h3 : in_interval a m n) : inequality a m n ↔ 1 ≤ a :=
sorry

end min_a_condition_l38_38096


namespace alex_cell_phone_cost_l38_38499

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.1
def extra_min_cost_per_minute : ℝ := 0.15
def text_messages_sent : ℕ := 150
def hours_talked : ℝ := 32
def included_hours : ℝ := 25

theorem alex_cell_phone_cost : base_cost 
  + (text_messages_sent * text_cost_per_message)
  + ((hours_talked - included_hours) * 60 * extra_min_cost_per_minute) = 98 := by
  sorry

end alex_cell_phone_cost_l38_38499


namespace pet_store_animals_left_l38_38281

def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15

def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

def birds_left : Nat := initial_birds - birds_sold
def puppies_left : Nat := initial_puppies - puppies_adopted
def cats_left : Nat := initial_cats
def spiders_left : Nat := initial_spiders - spiders_loose

def total_animals_left : Nat := birds_left + puppies_left + cats_left + spiders_left

theorem pet_store_animals_left : total_animals_left = 25 :=
by
  sorry

end pet_store_animals_left_l38_38281


namespace contractor_absent_days_l38_38269

variable (x y : ℕ)  -- Number of days worked and absent, both are natural numbers

-- Conditions from the problem
def total_days (x y : ℕ) : Prop := x + y = 30
def total_payment (x y : ℕ) : Prop := 25 * x - 75 * y / 10 = 360

-- Main statement
theorem contractor_absent_days (h1 : total_days x y) (h2 : total_payment x y) : y = 12 :=
by
  sorry

end contractor_absent_days_l38_38269


namespace range_of_m_l38_38622

noncomputable def f (a x : ℝ) := a * (x^2 + 1) + Real.log x

theorem range_of_m (a m : ℝ) (h₁ : a ∈ Set.Ioo (-4 : ℝ) (-2))
  (h₂ : ∀ x ∈ Set.Icc (1 : ℝ) (3), ma - f a x > a^2) : m ≤ -2 := 
sorry

end range_of_m_l38_38622


namespace complex_number_second_quadrant_l38_38016

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i * (1 + i)

-- Define a predicate to determine if a complex number is in the second quadrant
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main statement
theorem complex_number_second_quadrant : is_second_quadrant z := by
  sorry

end complex_number_second_quadrant_l38_38016


namespace area_of_figure_l38_38026

theorem area_of_figure : 
  ∀ (x y : ℝ), |3 * x + 4| + |4 * y - 3| ≤ 12 → area_of_rhombus = 24 := 
by
  sorry

end area_of_figure_l38_38026


namespace marigolds_sold_second_day_l38_38394

theorem marigolds_sold_second_day (x : ℕ) (h1 : 14 ≤ x)
  (h2 : 2 * x + 14 + x = 89) : x = 25 :=
by
  sorry

end marigolds_sold_second_day_l38_38394


namespace triangle_problem_l38_38807

-- Defining the conditions as Lean constructs
variable (a c : ℝ)
variable (b : ℝ := 3)
variable (cosB : ℝ := 1 / 3)
variable (dotProductBACBC : ℝ := 2)
variable (cosB_minus_C : ℝ := 23 / 27)

-- Define the problem as a theorem in Lean 4
theorem triangle_problem
  (h1 : a > c)
  (h2 : a * c * cosB = dotProductBACBC)
  (h3 : a^2 + c^2 = 13) :
  a = 3 ∧ c = 2 ∧ cosB_minus_C = 23 / 27 := by
  sorry

end triangle_problem_l38_38807


namespace polynomial_product_equals_expected_result_l38_38360

-- Define the polynomials
def polynomial_product (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- Define the expected result of the product
def expected_result (x : ℝ) : ℝ := x^3 + 1

-- The main theorem to prove
theorem polynomial_product_equals_expected_result (x : ℝ) : polynomial_product x = expected_result x :=
by
  -- Placeholder for the proof
  sorry

end polynomial_product_equals_expected_result_l38_38360


namespace trader_profit_percentage_l38_38918

theorem trader_profit_percentage
  (P : ℝ)
  (h1 : P > 0)
  (buy_price : ℝ := 0.80 * P)
  (sell_price : ℝ := 1.60 * P) :
  (sell_price - P) / P * 100 = 60 := 
by sorry

end trader_profit_percentage_l38_38918


namespace find_angle_D_l38_38561

variable (A B C D : ℝ)
variable (h1 : A + B = 180)
variable (h2 : C = D)
variable (h3 : C + 50 + 60 = 180)

theorem find_angle_D : D = 70 := by
  sorry

end find_angle_D_l38_38561


namespace last_two_digits_of_9_power_h_are_21_l38_38869

def a := 1
def b := 2^a
def c := 3^b
def d := 4^c
def e := 5^d
def f := 6^e
def g := 7^f
def h := 8^g

theorem last_two_digits_of_9_power_h_are_21 : (9^h) % 100 = 21 := by
  sorry

end last_two_digits_of_9_power_h_are_21_l38_38869


namespace length_CD_l38_38791

theorem length_CD (AB AC BD CD : ℝ) (hAB : AB = 2) (hAC : AC = 5) (hBD : BD = 6) :
    CD = 3 :=
by
  sorry

end length_CD_l38_38791


namespace number_of_pupils_l38_38633

theorem number_of_pupils (n : ℕ) : (83 - 63) / n = 1 / 2 → n = 40 :=
by
  intro h
  -- This is where the proof would go.
  sorry

end number_of_pupils_l38_38633


namespace probability_of_two_jacob_one_isaac_l38_38438

-- Definition of the problem conditions
def jacob_letters := 5
def isaac_letters := 5
def total_cards := 12
def cards_drawn := 3

-- Combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probability calculation
def probability_two_jacob_one_isaac : ℚ :=
  (C jacob_letters 2 * C isaac_letters 1 : ℚ) / (C total_cards cards_drawn : ℚ)

-- The statement of the problem
theorem probability_of_two_jacob_one_isaac :
  probability_two_jacob_one_isaac = 5 / 22 :=
  by sorry

end probability_of_two_jacob_one_isaac_l38_38438


namespace polynomial_evaluation_l38_38303

def p (x : ℝ) (a b c d : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_evaluation
  (a b c d : ℝ)
  (h1 : p 1 a b c d = 1993)
  (h2 : p 2 a b c d = 3986)
  (h3 : p 3 a b c d = 5979) :
  (1 / 4 : ℝ) * (p 11 a b c d + p (-7) a b c d) = 5233 := by
  sorry

end polynomial_evaluation_l38_38303


namespace trig_identity_proof_l38_38961

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l38_38961


namespace num_digits_c_l38_38221

theorem num_digits_c (a b c : ℕ) (ha : 10 ^ 2010 ≤ a ∧ a < 10 ^ 2011)
  (hb : 10 ^ 2011 ≤ b ∧ b < 10 ^ 2012)
  (h1 : a < b) (h2 : b < c)
  (div1 : ∃ k : ℕ, b + a = k * (b - a))
  (div2 : ∃ m : ℕ, c + b = m * (c - b)) :
  10 ^ 4 ≤ c ∧ c < 10 ^ 5 :=
sorry

end num_digits_c_l38_38221


namespace slices_with_both_toppings_l38_38346

theorem slices_with_both_toppings :
  ∀ (h p b : ℕ),
  (h + b = 9) ∧ (p + b = 12) ∧ (h + p + b = 15) → b = 6 :=
by
  sorry

end slices_with_both_toppings_l38_38346


namespace sum_of_numbers_l38_38145

theorem sum_of_numbers (x y : ℕ) (hx : 100 ≤ x ∧ x < 1000) (hy : 1000 ≤ y ∧ y < 10000) (h : 10000 * x + y = 12 * x * y) :
  x + y = 1083 :=
sorry

end sum_of_numbers_l38_38145


namespace part1_part2_l38_38419

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - 1| + |2 * x - a|

theorem part1 (x : ℝ) : (f x 2 < 2) ↔ (1/4 < x ∧ x < 5/4) := by
  sorry
  
theorem part2 (a : ℝ) (hx : ∀ x : ℝ, f x a ≥ 3 * a + 2) :
  (-3/2 ≤ a ∧ a ≤ -1/4) := by
  sorry

end part1_part2_l38_38419


namespace sum_of_operations_l38_38669

def operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem sum_of_operations : operation 12 5 + operation 8 3 = 174 := by
  sorry

end sum_of_operations_l38_38669


namespace num_ordered_pairs_l38_38927

theorem num_ordered_pairs :
  ∃ (m n : ℤ), (m * n ≥ 0) ∧ (m^3 + n^3 + 99 * m * n = 33^3) ∧ (35 = 35) :=
by
  sorry

end num_ordered_pairs_l38_38927


namespace average_speed_l38_38795

theorem average_speed (D : ℝ) (h1 : 0 < D) :
  let s1 := 60   -- speed from Q to B in miles per hour
  let s2 := 20   -- speed from B to C in miles per hour
  let d1 := 2 * D  -- distance from Q to B
  let d2 := D     -- distance from B to C
  let t1 := d1 / s1  -- time to travel from Q to B
  let t2 := d2 / s2  -- time to travel from B to C
  let total_distance := d1 + d2  -- total distance
  let total_time := t1 + t2   -- total time
  let average_speed := total_distance / total_time  -- average speed
  average_speed = 36 :=
by
  sorry

end average_speed_l38_38795


namespace seatingArrangementsAreSix_l38_38195

-- Define the number of seating arrangements for 4 people around a round table
def numSeatingArrangements : ℕ :=
  3 * 2 * 1 -- Following the condition that the narrator's position is fixed

-- The main theorem stating the number of different seating arrangements
theorem seatingArrangementsAreSix : numSeatingArrangements = 6 :=
  by
    -- This is equivalent to following the explanation of solution which is just multiplying the numbers
    sorry

end seatingArrangementsAreSix_l38_38195


namespace incorrect_positional_relationship_l38_38053

-- Definitions for the geometric relationships
def line := Type
def plane := Type

def parallel (l : line) (α : plane) : Prop := sorry
def perpendicular (l : line) (α : plane) : Prop := sorry
def subset (l : line) (α : plane) : Prop := sorry
def distinct (l m : line) : Prop := l ≠ m

-- Given conditions
variables (l m : line) (α : plane)

-- Theorem statement: prove that D is incorrect given the conditions
theorem incorrect_positional_relationship
  (h_distinct : distinct l m)
  (h_parallel_l_α : parallel l α)
  (h_parallel_m_α : parallel m α) :
  ¬ (parallel l m) :=
sorry

end incorrect_positional_relationship_l38_38053


namespace domain_of_lg_abs_x_minus_1_l38_38392

theorem domain_of_lg_abs_x_minus_1 (x : ℝ) : 
  (|x| - 1 > 0) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end domain_of_lg_abs_x_minus_1_l38_38392


namespace perimeter_of_fence_l38_38338

noncomputable def n : ℕ := 18
noncomputable def w : ℝ := 0.5
noncomputable def d : ℝ := 4

theorem perimeter_of_fence : 3 * ((n / 3 - 1) * d + n / 3 * w) = 69 := by
  sorry

end perimeter_of_fence_l38_38338


namespace johnny_hourly_wage_l38_38495

-- Definitions based on conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Theorem statement
theorem johnny_hourly_wage : total_earnings / hours_worked = 4.75 :=
by
  sorry

end johnny_hourly_wage_l38_38495


namespace average_age_of_cricket_team_l38_38488

theorem average_age_of_cricket_team :
  let captain_age := 28
  let ages_sum := 28 + (28 + 4) + (28 - 2) + (28 + 6)
  let remaining_players := 15 - 4
  let total_sum := ages_sum + remaining_players * (A - 1)
  let total_players := 15
  total_sum / total_players = 27.25 := 
by 
  sorry

end average_age_of_cricket_team_l38_38488


namespace child_ticket_cost_l38_38560

theorem child_ticket_cost 
    (total_people : ℕ) 
    (total_money_collected : ℤ) 
    (adult_ticket_price : ℤ) 
    (children_attended : ℕ) 
    (adults_count : ℕ) 
    (total_adult_cost : ℤ) 
    (total_child_cost : ℤ) 
    (c : ℤ)
    (total_people_eq : total_people = 22)
    (total_money_collected_eq : total_money_collected = 50)
    (adult_ticket_price_eq : adult_ticket_price = 8)
    (children_attended_eq : children_attended = 18)
    (adults_count_eq : adults_count = total_people - children_attended)
    (total_adult_cost_eq : total_adult_cost = adults_count * adult_ticket_price)
    (total_child_cost_eq : total_child_cost = children_attended * c)
    (money_collected_eq : total_money_collected = total_adult_cost + total_child_cost) 
  : c = 1 := 
  by
    sorry

end child_ticket_cost_l38_38560


namespace quadratic_equation_roots_transformation_l38_38734

theorem quadratic_equation_roots_transformation (α β : ℝ) 
  (h1 : 3 * α^2 + 7 * α + 4 = 0)
  (h2 : 3 * β^2 + 7 * β + 4 = 0) :
  ∃ y : ℝ, 21 * y^2 - 23 * y + 6 = 0 :=
sorry

end quadratic_equation_roots_transformation_l38_38734


namespace molecular_weight_K2Cr2O7_l38_38449

/--
K2Cr2O7 consists of:
- 2 K atoms
- 2 Cr atoms
- 7 O atoms

Atomic weights:
- K: 39.10 g/mol
- Cr: 52.00 g/mol
- O: 16.00 g/mol

We need to prove that the molecular weight of 4 moles of K2Cr2O7 is 1176.80 g/mol.
-/
theorem molecular_weight_K2Cr2O7 :
  let weight_K := 39.10
  let weight_Cr := 52.00
  let weight_O := 16.00
  let mol_weight_K2Cr2O7 := (2 * weight_K) + (2 * weight_Cr) + (7 * weight_O)
  (4 * mol_weight_K2Cr2O7) = 1176.80 :=
by
  sorry

end molecular_weight_K2Cr2O7_l38_38449


namespace number_of_red_balls_l38_38998

theorem number_of_red_balls (W R T : ℕ) (hW : W = 12) (h_freq : (R : ℝ) / (T : ℝ) = 0.25) (hT : T = W + R) : R = 4 :=
by
  sorry

end number_of_red_balls_l38_38998


namespace box_volume_l38_38515

theorem box_volume
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12)
  (h4 : l = h + 1) :
  l * w * h = 120 := 
sorry

end box_volume_l38_38515


namespace range_x0_of_perpendicular_bisector_intersects_x_axis_l38_38716

open Real

theorem range_x0_of_perpendicular_bisector_intersects_x_axis
  (A B : ℝ × ℝ) 
  (hA : (A.1^2 / 9) + (A.2^2 / 8) = 1)
  (hB : (B.1^2 / 9) + (B.2^2 / 8) = 1)
  (N : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hN : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hP : P.2 = 0) 
  (hl : P.1 = N.1 + (8 * N.1) / (9 * N.2) * N.2)
  : -1/3 < P.1 ∧ P.1 < 1/3 :=
sorry

end range_x0_of_perpendicular_bisector_intersects_x_axis_l38_38716


namespace algebraic_expression_value_l38_38933

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : 2*a^2 - 4*a + 2023 = 2025 :=
sorry

end algebraic_expression_value_l38_38933


namespace at_least_one_genuine_l38_38770

/-- Given 12 products, of which 10 are genuine and 2 are defective.
    If 3 products are randomly selected, then at least one of the selected products is a genuine product. -/
theorem at_least_one_genuine : 
  ∀ (products : Fin 12 → Prop), 
  (∃ n₁ n₂ : Fin 12, (n₁ ≠ n₂) ∧ 
                   (products n₁ = true) ∧ 
                   (products n₂ = true) ∧ 
                   (∃ n₁' n₂' : Fin 12, (n₁ ≠ n₁' ∧ n₂ ≠ n₂') ∧
                                         products n₁' = products n₂' = true ∧
                                         ∀ j : Fin 3, products j = true)) → 
  (∃ m : Fin 3, products m = true) :=
sorry

end at_least_one_genuine_l38_38770


namespace savings_calculation_l38_38714

def price_per_window : ℕ := 120
def discount_offer (n : ℕ) : ℕ := if n ≥ 10 then 2 else 0

def george_needs : ℕ := 9
def anne_needs : ℕ := 11

def cost (n : ℕ) : ℕ :=
  let free_windows := discount_offer n
  (n - free_windows) * price_per_window

theorem savings_calculation :
  let total_separate_cost := cost george_needs + cost anne_needs
  let total_windows := george_needs + anne_needs
  let total_cost_together := cost total_windows
  total_separate_cost - total_cost_together = 240 :=
by
  sorry

end savings_calculation_l38_38714


namespace man_work_rate_l38_38768

theorem man_work_rate (W : ℝ) (M S : ℝ)
  (h1 : (M + S) * 3 = W)
  (h2 : S * 5.25 = W) :
  M * 7 = W :=
by 
-- The proof steps will be filled in here.
sorry

end man_work_rate_l38_38768


namespace ratio_of_x_to_y_l38_38088

theorem ratio_of_x_to_y (x y : ℝ) (h : y = 0.20 * x) : x / y = 5 :=
by
  sorry

end ratio_of_x_to_y_l38_38088


namespace evaluate_expression_l38_38318

theorem evaluate_expression :
  (- (3 / 4 : ℚ)) / 3 * (- (2 / 5 : ℚ)) = 1 / 10 := 
by
  -- Here is where the proof would go
  sorry

end evaluate_expression_l38_38318


namespace polynomial_expansion_l38_38123

theorem polynomial_expansion (z : ℤ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  -- Provide a proof here
  sorry

end polynomial_expansion_l38_38123


namespace charlotte_one_way_journey_time_l38_38818

def charlotte_distance : ℕ := 60
def charlotte_speed : ℕ := 10

theorem charlotte_one_way_journey_time :
  charlotte_distance / charlotte_speed = 6 :=
by
  sorry

end charlotte_one_way_journey_time_l38_38818


namespace student_courses_last_year_l38_38457

variable (x : ℕ)
variable (courses_last_year : ℕ := x)
variable (avg_grade_last_year : ℕ := 100)
variable (courses_year_before : ℕ := 5)
variable (avg_grade_year_before : ℕ := 60)
variable (avg_grade_two_years : ℕ := 81)

theorem student_courses_last_year (h1 : avg_grade_last_year = 100)
                                   (h2 : courses_year_before = 5)
                                   (h3 : avg_grade_year_before = 60)
                                   (h4 : avg_grade_two_years = 81)
                                   (hc : ((5 * avg_grade_year_before) + (courses_last_year * avg_grade_last_year)) / (courses_year_before + courses_last_year) = avg_grade_two_years) :
                                   courses_last_year = 6 := by
  sorry

end student_courses_last_year_l38_38457


namespace find_number_l38_38541

theorem find_number (x : ℝ) (h : 0.26 * x = 93.6) : x = 360 := sorry

end find_number_l38_38541


namespace total_pigs_correct_l38_38480

def initial_pigs : Float := 64.0
def incoming_pigs : Float := 86.0
def total_pigs : Float := 150.0

theorem total_pigs_correct : initial_pigs + incoming_pigs = total_pigs := by 
  sorry

end total_pigs_correct_l38_38480


namespace sum_nonpositive_inequality_l38_38509

theorem sum_nonpositive_inequality (x : ℝ) : x + 5 ≤ 0 ↔ x + 5 ≤ 0 :=
by
  sorry

end sum_nonpositive_inequality_l38_38509


namespace prove_Praveen_present_age_l38_38936

-- Definitions based on the conditions identified in a)
def PraveenAge (P : ℝ) := P + 10 = 3 * (P - 3)

-- The equivalent proof problem statement
theorem prove_Praveen_present_age : ∃ P : ℝ, PraveenAge P ∧ P = 9.5 :=
by
  sorry

end prove_Praveen_present_age_l38_38936


namespace find_difference_l38_38445

-- Define the problem conditions in Lean
theorem find_difference (a b : ℕ) (hrelprime : Nat.gcd a b = 1)
                        (hpos : a > b) 
                        (hfrac : (a^3 - b^3) / (a - b)^3 = 73 / 3) :
    a - b = 3 :=
by
    sorry

end find_difference_l38_38445


namespace total_bananas_eq_l38_38769

def groups_of_bananas : ℕ := 2
def bananas_per_group : ℕ := 145

theorem total_bananas_eq : groups_of_bananas * bananas_per_group = 290 :=
by
  sorry

end total_bananas_eq_l38_38769


namespace colorful_family_total_children_l38_38945

theorem colorful_family_total_children (x : ℕ) (b : ℕ) :
  -- Initial equal number of white, blue, and striped children
  -- After some blue children become striped
  -- Total number of blue and white children was 10,
  -- Total number of white and striped children was 18
  -- We need to prove the total number of children is 21
  (x = 5) →
  (x + x = 10) →
  (10 + b = 18) →
  (3*x = 21) :=
by
  intros h1 h2 h3
  -- x initially represents the number of white, blue, and striped children
  -- We know x is 5 and satisfy the conditions
  sorry

end colorful_family_total_children_l38_38945


namespace arithmetic_geometric_seq_l38_38967

variable {a_n : ℕ → ℝ}
variable {a_1 a_3 a_5 a_6 a_11 : ℝ}

theorem arithmetic_geometric_seq (h₁ : a_1 * a_5 + 2 * a_3 * a_6 + a_1 * a_11 = 16) 
                                  (h₂ : a_1 * a_5 = a_3^2) 
                                  (h₃ : a_1 * a_11 = a_6^2) 
                                  (h₄ : a_3 > 0)
                                  (h₅ : a_6 > 0) : 
    a_3 + a_6 = 4 := 
by {
    sorry
}

end arithmetic_geometric_seq_l38_38967


namespace smallest_possible_N_l38_38799

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l38_38799


namespace xiaofang_time_l38_38717

-- Definitions
def overlap_time (t : ℕ) : Prop :=
  t - t / 12 = 40

def opposite_time (t : ℕ) : Prop :=
  t - t / 12 = 40

-- Theorem statement
theorem xiaofang_time :
  ∃ (x y : ℕ), 
    480 + x = 8 * 60 + 43 ∧
    840 + y = 2 * 60 + 43 ∧
    overlap_time x ∧
    opposite_time y ∧
    (y + 840 - (x + 480)) = 6 * 60 :=
by
  sorry

end xiaofang_time_l38_38717


namespace isosceles_triangle_large_angles_l38_38443

theorem isosceles_triangle_large_angles (y : ℝ) (h : 2 * y + 40 = 180) : y = 70 :=
by
  sorry

end isosceles_triangle_large_angles_l38_38443


namespace sum_of_cube_faces_l38_38264

theorem sum_of_cube_faces (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
    (h_eq_sum: (a * b * c) + (a * e * c) + (a * b * f) + (a * e * f) + (d * b * c) + (d * e * c) + (d * b * f) + (d * e * f) = 1089) :
    a + b + c + d + e + f = 31 := 
by
  sorry

end sum_of_cube_faces_l38_38264


namespace range_of_a_l38_38069

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l38_38069


namespace carnations_count_l38_38450

-- Define the conditions 
def vase_capacity : Nat := 9
def number_of_vases : Nat := 3
def number_of_roses : Nat := 23
def total_flowers : Nat := number_of_vases * vase_capacity

-- Define the number of carnations
def number_of_carnations : Nat := total_flowers - number_of_roses

-- Assertion that should be proved
theorem carnations_count : number_of_carnations = 4 := by
  sorry

end carnations_count_l38_38450


namespace count_players_studying_chemistry_l38_38181

theorem count_players_studying_chemistry :
  ∀ 
    (total_players : ℕ)
    (math_players : ℕ)
    (physics_players : ℕ)
    (math_and_physics_players : ℕ)
    (all_three_subjects_players : ℕ),
    total_players = 18 →
    math_players = 10 →
    physics_players = 6 →
    math_and_physics_players = 3 →
    all_three_subjects_players = 2 →
    (total_players - (math_players + physics_players - math_and_physics_players)) + all_three_subjects_players = 7 :=
by
  intros total_players math_players physics_players math_and_physics_players all_three_subjects_players
  sorry

end count_players_studying_chemistry_l38_38181


namespace domain_of_function_l38_38106

theorem domain_of_function :
  {x : ℝ | ∀ k : ℤ, 2 * x + (π / 4) ≠ k * π + (π / 2)}
  = {x : ℝ | ∀ k : ℤ, x ≠ (k * π / 2) + (π / 8)} :=
sorry

end domain_of_function_l38_38106


namespace simplify_expression_l38_38259

theorem simplify_expression : 
  2^345 - 3^4 * (3^2)^2 = 2^345 - 6561 := by
sorry

end simplify_expression_l38_38259


namespace pizza_toppings_l38_38521

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l38_38521


namespace distance_between_centers_l38_38033

theorem distance_between_centers (r1 r2 d x : ℝ) (h1 : r1 = 10) (h2 : r2 = 6) (h3 : d = 30) :
  x = 2 * Real.sqrt 229 := 
sorry

end distance_between_centers_l38_38033


namespace xiao_wang_programming_methods_l38_38614

theorem xiao_wang_programming_methods :
  ∃ (n : ℕ), n = 20 :=
by sorry

end xiao_wang_programming_methods_l38_38614


namespace total_stickers_l38_38362

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l38_38362


namespace smallest_nat_satisfies_conditions_l38_38265

theorem smallest_nat_satisfies_conditions : 
  ∃ x : ℕ, (∃ m : ℤ, x + 13 = 5 * m) ∧ (∃ n : ℤ, x - 13 = 6 * n) ∧ x = 37 := by
  sorry

end smallest_nat_satisfies_conditions_l38_38265


namespace fractions_integer_or_fractional_distinct_l38_38477

theorem fractions_integer_or_fractional_distinct (a b : Fin 6 → ℕ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_irreducible : ∀ i, Nat.gcd (a i) (b i) = 1)
  (h_sum_eq : (Finset.univ : Finset (Fin 6)).sum a = (Finset.univ : Finset (Fin 6)).sum b) :
  ¬ ∀ i j : Fin 6, i ≠ j → ((a i / b i = a j / b j) ∨ (a i % b i / b i = a j % b j / b j)) :=
sorry

end fractions_integer_or_fractional_distinct_l38_38477


namespace min_k_value_l38_38494

-- Definition of the problem's conditions
def remainder_condition (n k : ℕ) : Prop :=
  ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1

def in_range (x a b : ℕ) : Prop :=
  a < x ∧ x < b

-- The statement of the proof problem in Lean 4
theorem min_k_value (n k : ℕ) (h1 : remainder_condition n k) (hn_range : in_range n 2000 3000) :
  k = 9 :=
sorry

end min_k_value_l38_38494


namespace percentage_of_absent_students_l38_38451

theorem percentage_of_absent_students (total_students boys girls : ℕ) (fraction_boys_absent fraction_girls_absent : ℚ)
  (total_students_eq : total_students = 180)
  (boys_eq : boys = 120)
  (girls_eq : girls = 60)
  (fraction_boys_absent_eq : fraction_boys_absent = 1/6)
  (fraction_girls_absent_eq : fraction_girls_absent = 1/4) :
  let boys_absent := fraction_boys_absent * boys
  let girls_absent := fraction_girls_absent * girls
  let total_absent := boys_absent + girls_absent
  let absent_percentage := (total_absent / total_students) * 100
  abs (absent_percentage - 19) < 1 :=
by
  sorry

end percentage_of_absent_students_l38_38451


namespace correct_calculation_l38_38664

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := 
by 
  sorry

end correct_calculation_l38_38664


namespace contrapositive_example_l38_38371

variable {a : ℕ → ℝ}

theorem contrapositive_example 
  (h₁ : ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n ≤ a (n + 1)) → ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 ≥ a (n + 1) :=
by
  sorry

end contrapositive_example_l38_38371


namespace age_of_son_l38_38567

theorem age_of_son (S F : ℕ) (h1 : F = S + 28) (h2 : F + 2 = 2 * (S + 2)) : S = 26 := 
by
  -- skip the proof
  sorry

end age_of_son_l38_38567


namespace min_value_expression_l38_38990

open Real

theorem min_value_expression (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z)
    (h4: (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10):
    (x / y + y / z + z / x) * (y / x + z / y + x / z) = 25 :=
by
  sorry

end min_value_expression_l38_38990


namespace max_ab_value_l38_38363

noncomputable def max_ab (a b : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0 ∧ 2 * a + b = 1) then a * b else 0

theorem max_ab_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : 2 * a + b = 1) :
  max_ab a b = 1 / 8 := sorry

end max_ab_value_l38_38363


namespace number_of_nephews_l38_38605

def total_jellybeans : ℕ := 70
def jellybeans_per_child : ℕ := 14
def number_of_nieces : ℕ := 2

theorem number_of_nephews : total_jellybeans / jellybeans_per_child - number_of_nieces = 3 := by
  sorry

end number_of_nephews_l38_38605


namespace megan_homework_problems_l38_38722

theorem megan_homework_problems
  (finished_problems : ℕ)
  (pages_remaining : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ) :
  finished_problems = 26 →
  pages_remaining = 2 →
  problems_per_page = 7 →
  total_problems = finished_problems + (pages_remaining * problems_per_page) →
  total_problems = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end megan_homework_problems_l38_38722


namespace angle_CBD_is_48_degrees_l38_38672

theorem angle_CBD_is_48_degrees :
  ∀ (A B D C : Type) (α β γ δ : ℝ), 
    α = 28 ∧ β = 46 ∧ C ∈ [B, D] ∧ γ = 30 → 
    δ = 48 := 
by 
  sorry

end angle_CBD_is_48_degrees_l38_38672


namespace highest_percentage_without_car_l38_38952

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l38_38952


namespace base_equivalence_l38_38399

theorem base_equivalence : 
  ∀ (b : ℕ), (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 ↔ b = 10 := 
by
  sorry

end base_equivalence_l38_38399


namespace simplify_fraction_l38_38395

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l38_38395


namespace machine_A_production_is_4_l38_38600

noncomputable def machine_production (A : ℝ) (B : ℝ) (T_A : ℝ) (T_B : ℝ) := 
  (440 / A = T_A) ∧
  (440 / B = T_B) ∧
  (T_A = T_B + 10) ∧
  (B = 1.10 * A)

theorem machine_A_production_is_4 {A B T_A T_B : ℝ}
  (h : machine_production A B T_A T_B) : 
  A = 4 :=
by
  sorry

end machine_A_production_is_4_l38_38600


namespace total_cost_of_items_l38_38839

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l38_38839


namespace correct_calculation_l38_38403

variable {a b : ℝ}

theorem correct_calculation : 
  (2 * a^3 + 2 * a ≠ 2 * a^4) ∧
  ((a - 2 * b)^2 ≠ a^2 - 4 * b^2) ∧
  (-5 * (2 * a - b) ≠ -10 * a - 5 * b) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end correct_calculation_l38_38403


namespace equation_has_roots_l38_38405

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l38_38405


namespace min_elements_in_AS_l38_38352

theorem min_elements_in_AS (n : ℕ) (h : n ≥ 2) (S : Finset ℝ) (h_card : S.card = n) :
  ∃ (A_S : Finset ℝ), ∀ T : Finset ℝ, (∀ a b : ℝ, a ≠ b → a ∈ S → b ∈ S → (a + b) / 2 ∈ T) → 
  T.card ≥ 2 * n - 3 :=
sorry

end min_elements_in_AS_l38_38352


namespace probability_recurrence_relation_l38_38826

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l38_38826


namespace num_consecutive_sets_summing_to_90_l38_38015

-- Define the arithmetic sequence sum properties
theorem num_consecutive_sets_summing_to_90 : 
  ∃ n : ℕ, n ≥ 2 ∧
    ∃ (a : ℕ), 2 * a + n - 1 = 180 / n ∧
      (∃ k : ℕ, 
         k ≥ 2 ∧
         ∃ b : ℕ, 2 * b + k - 1 = 180 / k) ∧
      (∃ m : ℕ, 
         m ≥ 2 ∧ 
         ∃ c : ℕ, 2 * c + m - 1 = 180 / m) ∧
      (n = 3 ∨ n = 5 ∨ n = 9) :=
sorry

end num_consecutive_sets_summing_to_90_l38_38015


namespace moles_of_ammonia_combined_l38_38792

theorem moles_of_ammonia_combined (n_CO2 n_Urea n_NH3 : ℕ) (h1 : n_CO2 = 1) (h2 : n_Urea = 1) (h3 : n_Urea = n_CO2)
  (h4 : n_Urea = 2 * n_NH3): n_NH3 = 2 := 
by
  sorry

end moles_of_ammonia_combined_l38_38792


namespace jordan_buys_rice_l38_38493

variables (r l : ℝ)

theorem jordan_buys_rice
  (price_rice : ℝ := 1.20)
  (price_lentils : ℝ := 0.60)
  (total_pounds : ℝ := 30)
  (total_cost : ℝ := 27.00)
  (eq1 : r + l = total_pounds)
  (eq2 : price_rice * r + price_lentils * l = total_cost) :
  r = 15.0 :=
by
  sorry

end jordan_buys_rice_l38_38493


namespace area_difference_8_7_area_difference_9_8_l38_38348

-- Define the side lengths of the tablets
def side_length_7 : ℕ := 7
def side_length_8 : ℕ := 8
def side_length_9 : ℕ := 9

-- Define the areas of the tablets
def area_7 := side_length_7 * side_length_7
def area_8 := side_length_8 * side_length_8
def area_9 := side_length_9 * side_length_9

-- Prove the differences in area
theorem area_difference_8_7 : area_8 - area_7 = 15 := by sorry
theorem area_difference_9_8 : area_9 - area_8 = 17 := by sorry

end area_difference_8_7_area_difference_9_8_l38_38348


namespace ratio_angela_jacob_l38_38418

-- Definitions for the conditions
def deans_insects := 30
def jacobs_insects := 5 * deans_insects
def angelas_insects := 75

-- The proof statement proving the ratio
theorem ratio_angela_jacob : angelas_insects / jacobs_insects = 1 / 2 :=
by
  -- Sorry is used here to indicate that the proof is skipped
  sorry

end ratio_angela_jacob_l38_38418


namespace exponent_rule_example_l38_38246

theorem exponent_rule_example : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end exponent_rule_example_l38_38246


namespace power_add_one_eq_twice_l38_38907

theorem power_add_one_eq_twice (a b : ℕ) (h : 2^a = b) : 2^(a + 1) = 2 * b := by
  sorry

end power_add_one_eq_twice_l38_38907


namespace intersection_P_Q_eq_Q_l38_38510

def P : Set ℝ := { x | x < 2 }
def Q : Set ℝ := { x | x^2 ≤ 1 }

theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
sorry

end intersection_P_Q_eq_Q_l38_38510


namespace peter_present_age_l38_38744

def age_problem (P J : ℕ) : Prop :=
  J = P + 12 ∧ P - 10 = (1 / 3 : ℚ) * (J - 10)

theorem peter_present_age : ∃ (P : ℕ), ∃ (J : ℕ), age_problem P J ∧ P = 16 :=
by {
  -- Add the proof here, which is not required
  sorry
}

end peter_present_age_l38_38744


namespace Nancy_antacid_consumption_l38_38398

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end Nancy_antacid_consumption_l38_38398


namespace tan_half_angle_sin_cos_expression_l38_38426

-- Proof Problem 1: If α is an angle in the third quadrant and sin α = -5/13, then tan (α / 2) = -5.
theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -5 := 
by 
  sorry

-- Proof Problem 2: If tan α = 2, then sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5.
theorem sin_cos_expression (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 :=
by 
  sorry

end tan_half_angle_sin_cos_expression_l38_38426


namespace sum_of_a_b_l38_38146

-- Definitions for the given conditions
def geom_series_sum (a : ℤ) (n : ℕ) : ℤ := 2^n + a
def arith_series_sum (b : ℤ) (n : ℕ) : ℤ := n^2 - 2*n + b

-- Theorem statement
theorem sum_of_a_b (a b : ℤ) (h1 : ∀ n, geom_series_sum a n = 2^n + a)
  (h2 : ∀ n, arith_series_sum b n = n^2 - 2*n + b) :
  a + b = -1 :=
sorry

end sum_of_a_b_l38_38146


namespace total_revenue_correct_l38_38175

-- Define the conditions
def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sold_sneakers : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sold_sandals : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.4
def pairs_sold_boots : ℕ := 11

-- Compute discounted prices
def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (original_price * discount)

-- Compute revenue from each type of shoe
def revenue (price : ℝ) (pairs_sold : ℕ) : ℝ :=
  price * (pairs_sold : ℝ)

open Real

-- Main statement to prove
theorem total_revenue_correct : 
  revenue (discounted_price original_price_sneakers discount_sneakers) pairs_sold_sneakers + 
  revenue (discounted_price original_price_sandals discount_sandals) pairs_sold_sandals + 
  revenue (discounted_price original_price_boots discount_boots) pairs_sold_boots = 1068 := 
by
  sorry

end total_revenue_correct_l38_38175


namespace infinite_geometric_series_common_ratio_l38_38215

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l38_38215


namespace Yihana_uphill_walking_time_l38_38551

theorem Yihana_uphill_walking_time :
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  t_total = 5 :=
by
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  show t_total = 5
  sorry

end Yihana_uphill_walking_time_l38_38551


namespace find_constant_b_l38_38599

theorem find_constant_b 
  (a b c : ℝ)
  (h1 : 3 * a = 9) 
  (h2 : (-2 * a + 3 * b) = -5) 
  : b = 1 / 3 :=
by 
  have h_a : a = 3 := by linarith
  
  have h_b : -2 * 3 + 3 * b = -5 := by linarith [h2]
  
  linarith

end find_constant_b_l38_38599


namespace original_number_l38_38472

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end original_number_l38_38472


namespace alpha_sufficient_but_not_necessary_condition_of_beta_l38_38584
open Classical

variable (x : ℝ)
def α := x = -1
def β := x ≤ 0

theorem alpha_sufficient_but_not_necessary_condition_of_beta :
  (α x → β x) ∧ ¬(β x → α x) :=
by
  sorry

end alpha_sufficient_but_not_necessary_condition_of_beta_l38_38584


namespace years_of_school_eq_13_l38_38050

/-- Conditions definitions -/
def cost_per_semester : ℕ := 20000
def semesters_per_year : ℕ := 2
def total_cost : ℕ := 520000

/-- Derived definitions from conditions -/
def cost_per_year := cost_per_semester * semesters_per_year
def number_of_years := total_cost / cost_per_year

/-- Proof that number of years equals 13 given the conditions -/
theorem years_of_school_eq_13 : number_of_years = 13 :=
by sorry

end years_of_school_eq_13_l38_38050


namespace complement_intersection_l38_38323

open Set

variable (U : Set ℤ) (A B : Set ℤ)

theorem complement_intersection (hU : U = univ)
                               (hA : A = {3, 4})
                               (h_union : A ∪ B = {1, 2, 3, 4}) :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end complement_intersection_l38_38323


namespace problem_inequality_l38_38407

open Real

theorem problem_inequality 
  (p q r x y theta: ℝ) :
  p * x ^ (q - y) + q * x ^ (r - y) + r * x ^ (y - theta)  ≥ p + q + r :=
sorry

end problem_inequality_l38_38407


namespace range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l38_38287

open Set

noncomputable def A (a : ℝ) : Set ℝ := { x : ℝ | a - 1 < x ∧ x < 2 * a + 1 }
def B : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

theorem range_of_a_union_B_eq_A (a : ℝ) :
  (A a ∪ B) = A a ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

theorem range_of_a_inter_B_eq_empty (a : ℝ) :
  (A a ∩ B) = ∅ ↔ (a ≤ - 1 / 2 ∨ 2 ≤ a) := by
  sorry

end range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l38_38287


namespace total_cost_is_130_l38_38063

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end total_cost_is_130_l38_38063


namespace find_m_n_sum_l38_38274

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end find_m_n_sum_l38_38274


namespace find_smallest_n_l38_38703

theorem find_smallest_n : ∃ n : ℕ, (n - 4)^3 > (n^3 / 2) ∧ ∀ m : ℕ, m < n → (m - 4)^3 ≤ (m^3 / 2) :=
by
  sorry

end find_smallest_n_l38_38703


namespace triangle_properties_l38_38801

-- Define the sides of the triangle
def side1 : ℕ := 8
def side2 : ℕ := 15
def hypotenuse : ℕ := 17

-- Using the Pythagorean theorem to assert it is a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Calculate the area of the right triangle
def triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Calculate the perimeter of the triangle
def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_properties :
  let a := side1
  let b := side2
  let c := hypotenuse
  is_right_triangle a b c →
  triangle_area a b = 60 ∧ triangle_perimeter a b c = 40 := by
  intros h
  sorry

end triangle_properties_l38_38801


namespace side_length_of_octagon_l38_38092

-- Define the conditions
def is_octagon (n : ℕ) := n = 8
def perimeter (p : ℕ) := p = 72

-- Define the problem statement
theorem side_length_of_octagon (n p l : ℕ) 
  (h1 : is_octagon n) 
  (h2 : perimeter p) 
  (h3 : p / n = l) :
  l = 9 := 
  sorry

end side_length_of_octagon_l38_38092


namespace problem_l38_38821

variable (x : ℝ)

theorem problem (A B : ℝ) 
  (h : (A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3))): 
  A + B = 15 := by
  sorry

end problem_l38_38821


namespace sum_of_reciprocals_ineq_l38_38712

theorem sum_of_reciprocals_ineq (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a ^ 2 - 4 * a + 11)) + 
  (1 / (5 * b ^ 2 - 4 * b + 11)) + 
  (1 / (5 * c ^ 2 - 4 * c + 11)) ≤ 
  (1 / 4) := 
by {
  sorry
}

end sum_of_reciprocals_ineq_l38_38712


namespace problem_l38_38056

def f (x : ℤ) : ℤ := 7 * x - 3

theorem problem : f (f (f 3)) = 858 := by
  sorry

end problem_l38_38056


namespace nth_term_correct_l38_38232

noncomputable def nth_term (a b : ℝ) (n : ℕ) : ℝ :=
  (-1 : ℝ)^n * (2 * n - 1) * b / a^n

theorem nth_term_correct (a b : ℝ) (n : ℕ) (h : 0 < a) : 
  nth_term a b n = (-1 : ℝ)^↑n * (2 * n - 1) * b / a^n :=
by sorry

end nth_term_correct_l38_38232


namespace remainder_when_divided_by_44_l38_38216

theorem remainder_when_divided_by_44 (N Q R : ℕ) :
  (N = 44 * 432 + R) ∧ (N = 39 * Q + 15) → R = 0 :=
by
  sorry

end remainder_when_divided_by_44_l38_38216


namespace rewrite_expression_l38_38468

theorem rewrite_expression (k : ℝ) :
  ∃ d r s : ℝ, (8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ (r + s = 14.75) := 
sorry

end rewrite_expression_l38_38468


namespace coefficient_x3_l38_38152

-- Define the binomial coefficient
def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

noncomputable def coefficient_x3_term : Nat :=
  binomial_coefficient 25 3

theorem coefficient_x3 : coefficient_x3_term = 2300 :=
by
  unfold coefficient_x3_term
  unfold binomial_coefficient
  -- Here, one would normally provide the proof steps, but we're adding sorry to skip
  sorry

end coefficient_x3_l38_38152


namespace not_mysterious_diff_consecutive_odd_l38_38516

/-- A mysterious number is defined as the difference of squares of two consecutive even numbers. --/
def is_mysterious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 2)^2 - (2 * k)^2

/-- The difference of the squares of two consecutive odd numbers. --/
def diff_squares_consecutive_odd (k : ℤ) : ℤ :=
  (2 * k + 1)^2 - (2 * k - 1)^2

/-- Prove that the difference of squares of two consecutive odd numbers is not a mysterious number. --/
theorem not_mysterious_diff_consecutive_odd (k : ℤ) : ¬ is_mysterious (Int.natAbs (diff_squares_consecutive_odd k)) :=
by
  sorry

end not_mysterious_diff_consecutive_odd_l38_38516


namespace sqrt_sum_inequality_l38_38671

open Real

theorem sqrt_sum_inequality (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ sqrt 21 :=
sorry

end sqrt_sum_inequality_l38_38671


namespace day_crew_fraction_correct_l38_38586

variable (D Wd : ℕ) -- D = number of boxes loaded by each worker on the day crew, Wd = number of workers on the day crew

-- fraction of all boxes loaded by day crew
def fraction_loaded_by_day_crew (D Wd : ℕ) : ℚ :=
  (D * Wd) / (D * Wd + (3 / 4 * D) * (2 / 3 * Wd))

theorem day_crew_fraction_correct (h1 : D > 0) (h2 : Wd > 0) :
  fraction_loaded_by_day_crew D Wd = 2 / 3 := by
  sorry

end day_crew_fraction_correct_l38_38586


namespace ratio_a_b_is_zero_l38_38603

-- Setting up the conditions
variables (a y b : ℝ)
variable (d : ℝ)
-- Condition for arithmetic sequence
axiom h1 : a + d = y
axiom h2 : y + d = b
axiom h3 : b + d = 3 * y

-- The Lean statement to prove
theorem ratio_a_b_is_zero (h1 : a + d = y) (h2 : y + d = b) (h3 : b + d = 3 * y) : a / b = 0 :=
sorry

end ratio_a_b_is_zero_l38_38603


namespace quadratic_has_distinct_real_roots_expression_value_l38_38341

variable (x m : ℝ)

-- Condition: Quadratic equation
def quadratic_eq := (x^2 - 2 * (m - 1) * x - m * (m + 2) = 0)

-- Prove that the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (m : ℝ) : 
  ∃ a b : ℝ, a ≠ b ∧ quadratic_eq a m ∧ quadratic_eq b m :=
by
  sorry

-- Given that x = -2 is a root, prove that 2018 - 3(m-1)^2 = 2015
theorem expression_value (m : ℝ) (h : quadratic_eq (-2) m) : 
  2018 - 3 * (m - 1)^2 = 2015 :=
by
  sorry

end quadratic_has_distinct_real_roots_expression_value_l38_38341


namespace find_PS_eq_13point625_l38_38906

theorem find_PS_eq_13point625 (PQ PR QR : ℝ) (h : ℝ) (QS SR : ℝ)
  (h_QS : QS^2 = 225 - h^2)
  (h_SR : SR^2 = 400 - h^2)
  (h_ratio : QS / SR = 3 / 7) :
  PS = 13.625 :=
by
  sorry

end find_PS_eq_13point625_l38_38906


namespace factor_expression_value_l38_38006

theorem factor_expression_value :
  ∃ (k m n : ℕ), 
    k > 1 ∧ m > 1 ∧ n > 1 ∧ 
    k ≤ 60 ∧ m ≤ 35 ∧ n ≤ 20 ∧ 
    (2^k + 3^m + k^3 * m^n - n = 43) :=
by
  sorry

end factor_expression_value_l38_38006


namespace largest_angle_in_consecutive_integer_hexagon_l38_38836

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l38_38836


namespace find_alpha_l38_38524

noncomputable def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * (a 2 / a 1)

-- Given that {a_n} is a geometric sequence,
-- a_1 and a_8 are roots of the equation
-- x^2 - 2x * sin(alpha) - √3 * sin(alpha) = 0,
-- and (a_1 + a_8)^2 = 2 * a_3 * a_6 + 6,
-- prove that alpha = π / 3.
theorem find_alpha :
  ∃ α : ℝ,
  (∀ (a : ℕ → ℝ), isGeometricSequence a ∧ 
  (∃ (a1 a8 : ℝ), 
    (a1 + a8)^2 = 2 * a 3 * a 6 + 6 ∧
    a1 + a8 = 2 * Real.sin α ∧
    a1 * a8 = - Real.sqrt 3 * Real.sin α)) →
  α = Real.pi / 3 :=
by 
  sorry

end find_alpha_l38_38524


namespace find_f_1_minus_a_l38_38143

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem find_f_1_minus_a 
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_period : periodic_function f 2)
  (h_value : ∃ a : ℝ, f (1 + a) = 1) :
  ∃ a : ℝ, f (1 - a) = -1 :=
by
  sorry

end find_f_1_minus_a_l38_38143


namespace minimum_value_expression_l38_38518

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  4 * a ^ 3 + 8 * b ^ 3 + 27 * c ^ 3 + 64 * d ^ 3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

end minimum_value_expression_l38_38518


namespace howard_groups_l38_38761

theorem howard_groups :
  (18 : ℕ) / (24 / 4) = 3 := sorry

end howard_groups_l38_38761


namespace expression_evaluation_l38_38334

theorem expression_evaluation :
  1 - (2 - (3 - 4 - (5 - 6))) = -1 :=
sorry

end expression_evaluation_l38_38334


namespace f_inequality_l38_38219

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l38_38219


namespace fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l38_38589

theorem fraction_sum_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ (3 * 5 * n * (a + b) = a * b) :=
sorry

theorem fraction_sum_of_equal_reciprocals (n : ℕ) : 
  ∃ a : ℕ, 3 * 5 * n * 2 = a * a ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

theorem fraction_difference_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ 3 * 5 * n * (a - b) = a * b :=
sorry

end fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l38_38589


namespace number_of_pairs_l38_38155

theorem number_of_pairs (x y : ℤ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000) :
  (x^2 + y^2) % 7 = 0 → (∃ n : ℕ, n = 20164) :=
by {
  sorry
}

end number_of_pairs_l38_38155


namespace min_value_a_plus_b_l38_38957

open Real

theorem min_value_a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 1 / a + 2 / b = 1) :
  a + b = 3 + 2 * sqrt 2 :=
sorry

end min_value_a_plus_b_l38_38957


namespace circle_tangent_independence_l38_38133

noncomputable def e1 (r : ℝ) (β : ℝ) := r * Real.tan β
noncomputable def e2 (r : ℝ) (α : ℝ) := r * Real.tan α
noncomputable def e3 (r : ℝ) (β α : ℝ) := r * Real.tan (β - α)

theorem circle_tangent_independence 
  (O : ℝ) (r β α : ℝ) (hβ : β < π / 2) (hα : 0 < α) (hαβ : α < β) :
  (e1 r β) * (e2 r α) * (e3 r β α) / ((e1 r β) - (e2 r α) - (e3 r β α)) = r^2 :=
by
  sorry

end circle_tangent_independence_l38_38133


namespace cos_frac_less_sin_frac_l38_38197

theorem cos_frac_less_sin_frac : 
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  a < b :=
by
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  sorry -- proof skipped

end cos_frac_less_sin_frac_l38_38197


namespace fraction_of_original_water_after_four_replacements_l38_38386

-- Define the initial condition and process
def initial_water_volume : ℚ := 10
def initial_alcohol_volume : ℚ := 10
def initial_total_volume : ℚ := initial_water_volume + initial_alcohol_volume

def fraction_remaining_after_removal (fraction_remaining : ℚ) : ℚ :=
  fraction_remaining * (initial_total_volume - 5) / initial_total_volume

-- Define the function counting the iterations process
def fraction_after_replacements (n : ℕ) (fraction_remaining : ℚ) : ℚ :=
  Nat.iterate fraction_remaining_after_removal n fraction_remaining

-- We have 4 replacements, start with 1 (because initially half of tank is water, 
-- fraction is 1 means we start with all original water)
def fraction_of_original_water_remaining : ℚ := (fraction_after_replacements 4 1)

-- Our goal in proof form
theorem fraction_of_original_water_after_four_replacements :
  fraction_of_original_water_remaining = (81 / 256) := by
  sorry

end fraction_of_original_water_after_four_replacements_l38_38386


namespace min_value_frac_l38_38601

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : a + b = 1) : 
  (1 / a) + (4 / b) ≥ 9 :=
by sorry

end min_value_frac_l38_38601


namespace fraction_powers_sum_l38_38875

theorem fraction_powers_sum : 
  ( (5:ℚ) / (3:ℚ) )^6 + ( (2:ℚ) / (3:ℚ) )^6 = (15689:ℚ) / (729:ℚ) :=
by
  sorry

end fraction_powers_sum_l38_38875


namespace rachel_age_when_emily_half_her_age_l38_38532

theorem rachel_age_when_emily_half_her_age (emily_current_age rachel_current_age : ℕ) 
  (h1 : emily_current_age = 20) 
  (h2 : rachel_current_age = 24) 
  (age_difference : ℕ) 
  (h3 : rachel_current_age - emily_current_age = age_difference) 
  (emily_age_when_half : ℕ) 
  (rachel_age_when_half : ℕ) 
  (h4 : emily_age_when_half = rachel_age_when_half / 2)
  (h5 : rachel_age_when_half = emily_age_when_half + age_difference) :
  rachel_age_when_half = 8 :=
by
  sorry

end rachel_age_when_emily_half_her_age_l38_38532


namespace probability_top_card_is_king_or_queen_l38_38325

-- Defining the basic entities of the problem
def standard_deck_size := 52
def ranks := 13
def suits := 4
def number_of_kings := 4
def number_of_queens := 4
def number_of_kings_and_queens := number_of_kings + number_of_queens

-- Statement: Calculating the probability that the top card is either a King or a Queen
theorem probability_top_card_is_king_or_queen :
  (number_of_kings_and_queens : ℚ) / standard_deck_size = 2 / 13 := by
  -- Skipping the proof for now
  sorry

end probability_top_card_is_king_or_queen_l38_38325


namespace speed_of_stream_l38_38199

theorem speed_of_stream
  (b s : ℝ)
  (H1 : 120 = 2 * (b + s))
  (H2 : 60 = 2 * (b - s)) :
  s = 15 :=
by
  sorry

end speed_of_stream_l38_38199


namespace function_characterization_l38_38755

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  (∀ x : ℝ, 0 ≤ x → f x = if x < 2 then 2 / (2 - x) else 0) := sorry

end function_characterization_l38_38755


namespace spherical_to_rectangular_example_l38_38830

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 5 (3 * Real.pi / 2) (Real.pi / 3) = (0, -5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  simp [spherical_to_rectangular, Real.sin, Real.cos]
  sorry

end spherical_to_rectangular_example_l38_38830


namespace last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l38_38057

-- Define the cycle period used in the problem
def cycle_period_2 := [2, 4, 8, 6]
def cycle_period_3 := [3, 9, 7, 1]
def cycle_period_7 := [7, 9, 3, 1]

-- Define a function to get the last digit from the cycle for given n
def last_digit_from_cycle (cycle : List ℕ) (n : ℕ) : ℕ :=
  let cycle_length := cycle.length
  cycle.get! ((n % cycle_length) - 1)

-- Problem statements
theorem last_digit_2_pow_1000 : last_digit_from_cycle cycle_period_2 1000 = 6 := sorry
theorem last_digit_3_pow_1000 : last_digit_from_cycle cycle_period_3 1000 = 1 := sorry
theorem last_digit_7_pow_1000 : last_digit_from_cycle cycle_period_7 1000 = 1 := sorry

end last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l38_38057


namespace find_n_values_l38_38085

theorem find_n_values (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 91 = k^2) : n = 9 ∨ n = 10 :=
sorry

end find_n_values_l38_38085


namespace loss_percentage_l38_38615

/--
A man sells a car to his friend at a certain loss percentage. The friend then sells it 
for Rs. 54000 and gains 20%. The original cost price of the car was Rs. 52941.17647058824.
Prove that the loss percentage when the man sold the car to his friend was 15%.
-/
theorem loss_percentage (CP SP_2 : ℝ) (gain_percent : ℝ) (h_CP : CP = 52941.17647058824) 
(h_SP2 : SP_2 = 54000) (h_gain : gain_percent = 20) : (CP - SP_2 / (1 + gain_percent / 100)) / CP * 100 = 15 := by
  sorry

end loss_percentage_l38_38615


namespace fraction_simplification_l38_38940

theorem fraction_simplification (x : ℝ) (h : x = 0.5 * 106) : 18 / x = 18 / 53 := by
  rw [h]
  norm_num

end fraction_simplification_l38_38940


namespace ratio_steel_iron_is_5_to_2_l38_38989

-- Definitions based on the given conditions
def amount_steel : ℕ := 35
def amount_iron : ℕ := 14

-- Main statement
theorem ratio_steel_iron_is_5_to_2 :
  (amount_steel / Nat.gcd amount_steel amount_iron) = 5 ∧
  (amount_iron / Nat.gcd amount_steel amount_iron) = 2 :=
by
  sorry

end ratio_steel_iron_is_5_to_2_l38_38989


namespace solution_set_for_inequality_l38_38427

def f (x : ℝ) : ℝ := x^3 + x

theorem solution_set_for_inequality {a : ℝ} (h : -2 < a ∧ a < 2) :
  f a + f (a^2 - 2) < 0 ↔ -2 < a ∧ a < 0 ∨ 0 < a ∧ a < 1 := sorry

end solution_set_for_inequality_l38_38427


namespace find_sum_of_a_and_d_l38_38314

theorem find_sum_of_a_and_d 
  {a b c d : ℝ} 
  (h1 : ab + ac + bd + cd = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 :=
sorry

end find_sum_of_a_and_d_l38_38314


namespace vector_eq_to_slope_intercept_form_l38_38144

theorem vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) + 5 * (y - 1)) = 0 → y = -(2 / 5) * x + 13 / 5 := 
by 
  intros x y h
  sorry

end vector_eq_to_slope_intercept_form_l38_38144


namespace crayons_left_l38_38279

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l38_38279


namespace initial_velocity_is_three_l38_38377

-- Define the displacement function s(t)
def s (t : ℝ) : ℝ := 3 * t - t ^ 2

-- Define the initial time condition
def initial_time : ℝ := 0

-- State the main theorem about the initial velocity
theorem initial_velocity_is_three : (deriv s) initial_time = 3 :=
by
  sorry

end initial_velocity_is_three_l38_38377


namespace union_of_P_and_Q_l38_38309

def P : Set ℝ := { x | |x| ≥ 3 }
def Q : Set ℝ := { y | ∃ x, y = 2^x - 1 }

theorem union_of_P_and_Q : P ∪ Q = { y | y ≤ -3 ∨ y > -1 } := by
  sorry

end union_of_P_and_Q_l38_38309


namespace geometric_series_sum_l38_38261

theorem geometric_series_sum :
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 5 → 
  r = -1 / 5 → 
  n = 6 →
  (a - a * r^n) / (1 - r) = 1562 / 9375 :=
by 
  intro a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l38_38261


namespace recreation_percentage_correct_l38_38649

noncomputable def recreation_percentage (W : ℝ) : ℝ :=
  let recreation_two_weeks_ago := 0.25 * W
  let wages_last_week := 0.95 * W
  let recreation_last_week := 0.35 * (0.95 * W)
  let wages_this_week := 0.95 * W * 0.85
  let recreation_this_week := 0.45 * (0.95 * W * 0.85)
  (recreation_this_week / recreation_two_weeks_ago) * 100

theorem recreation_percentage_correct (W : ℝ) : recreation_percentage W = 145.35 :=
by
  sorry

end recreation_percentage_correct_l38_38649


namespace four_number_theorem_l38_38196

theorem four_number_theorem (a b c d : ℕ) (H : a * b = c * d) (Ha : 0 < a) (Hb : 0 < b) (Hc : 0 < c) (Hd : 0 < d) : 
  ∃ (p q r s : ℕ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r :=
by
  sorry

end four_number_theorem_l38_38196


namespace distance_from_center_to_line_of_tangent_circle_l38_38178

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l38_38178


namespace carmen_candle_burn_time_l38_38747

theorem carmen_candle_burn_time 
  (burn_time_first_scenario : ℕ)
  (nights_per_candle : ℕ)
  (total_candles_second_scenario : ℕ)
  (total_nights_second_scenario : ℕ)
  (h1 : burn_time_first_scenario = 1)
  (h2 : nights_per_candle = 8)
  (h3 : total_candles_second_scenario = 6)
  (h4 : total_nights_second_scenario = 24) :
  (total_candles_second_scenario * nights_per_candle) / total_nights_second_scenario = 2 :=
by
  sorry

end carmen_candle_burn_time_l38_38747


namespace Q_has_negative_and_potentially_positive_roots_l38_38002

def Q (x : ℝ) : ℝ := x^7 - 4 * x^6 + 2 * x^5 - 9 * x^3 + 2 * x + 16

theorem Q_has_negative_and_potentially_positive_roots :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧ (∃ y : ℝ, y > 0 ∧ Q y = 0 ∨ ∀ z : ℝ, Q z > 0) :=
by
  sorry

end Q_has_negative_and_potentially_positive_roots_l38_38002


namespace total_flour_required_l38_38046

-- Definitions specified based on the given conditions
def flour_already_put_in : ℕ := 10
def flour_needed : ℕ := 2

-- Lean 4 statement to prove the total amount of flour required by the recipe
theorem total_flour_required : (flour_already_put_in + flour_needed) = 12 :=
by
  sorry

end total_flour_required_l38_38046


namespace trust_meteorologist_l38_38374

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l38_38374


namespace decreasing_interval_l38_38428

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem decreasing_interval : ∀ x : ℝ, (-2 < x ∧ x < 0) → (deriv f x < 0) := 
by
  sorry

end decreasing_interval_l38_38428


namespace select_subset_divisible_by_n_l38_38052

theorem select_subset_divisible_by_n (n : ℕ) (h : n > 0) (l : List ℤ) (hl : l.length = 2 * n - 1) :
  ∃ s : Finset ℤ, s.card = n ∧ (s.sum id) % n = 0 := 
sorry

end select_subset_divisible_by_n_l38_38052


namespace lines_skew_l38_38704

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 3 + 2 * t, b + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 4 + 3 * u, 1 + 2 * u)

theorem lines_skew (b : ℝ) : 
  ¬ ∃ t u : ℝ, line1 b t = line2 u ↔ b ≠ 4 := 
sorry

end lines_skew_l38_38704


namespace bond_selling_price_l38_38293

theorem bond_selling_price
    (face_value : ℝ)
    (interest_rate_face : ℝ)
    (interest_rate_selling : ℝ)
    (interest : ℝ)
    (selling_price : ℝ)
    (h1 : face_value = 5000)
    (h2 : interest_rate_face = 0.07)
    (h3 : interest_rate_selling = 0.065)
    (h4 : interest = face_value * interest_rate_face)
    (h5 : interest = selling_price * interest_rate_selling) :
  selling_price = 5384.62 :=
sorry

end bond_selling_price_l38_38293


namespace dave_pieces_l38_38031

theorem dave_pieces (boxes_bought : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) 
  (h₁ : boxes_bought = 12) (h₂ : boxes_given = 5) (h₃ : pieces_per_box = 3) : 
  boxes_bought - boxes_given * pieces_per_box = 21 :=
by
  sorry

end dave_pieces_l38_38031


namespace trapezoid_total_area_l38_38977

/-- 
Given a trapezoid with side lengths 4, 6, 8, and 10, where sides 4 and 8 are used as parallel bases, 
prove that the total area of the trapezoid in all possible configurations is 48√2.
-/
theorem trapezoid_total_area : 
  let a := 4
  let b := 8
  let c := 6
  let d := 10
  let h := 4 * Real.sqrt 2
  let Area := (1 / 2) * (a + b) * h
  (Area + Area) = 48 * Real.sqrt 2 :=
by 
  sorry

end trapezoid_total_area_l38_38977


namespace fraction_area_of_triangles_l38_38189

theorem fraction_area_of_triangles 
  (base_PQR : ℝ) (height_PQR : ℝ)
  (base_XYZ : ℝ) (height_XYZ : ℝ)
  (h_base_PQR : base_PQR = 3)
  (h_height_PQR : height_PQR = 2)
  (h_base_XYZ : base_XYZ = 6)
  (h_height_XYZ : height_XYZ = 3) :
  (1/2 * base_PQR * height_PQR) / (1/2 * base_XYZ * height_XYZ) = 1 / 3 :=
by
  sorry

end fraction_area_of_triangles_l38_38189


namespace production_difference_l38_38007

theorem production_difference (w t : ℕ) (h1 : w = 3 * t) :
  (w * t) - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end production_difference_l38_38007


namespace number_of_black_bears_l38_38783

-- Definitions of conditions
def brown_bears := 15
def white_bears := 24
def total_bears := 66

-- The proof statement
theorem number_of_black_bears : (total_bears - (brown_bears + white_bears) = 27) := by
  sorry

end number_of_black_bears_l38_38783


namespace vasya_max_triangles_l38_38700

theorem vasya_max_triangles (n : ℕ) (h1 : n = 100)
  (h2 : ∀ (a b c : ℕ), a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :
  ∃ (t : ℕ), t = n := 
sorry

end vasya_max_triangles_l38_38700


namespace arithmetic_mean_of_pq_is_10_l38_38497

variable (p q r : ℝ)

theorem arithmetic_mean_of_pq_is_10
  (H_mean_qr : (q + r) / 2 = 20)
  (H_r_minus_p : r - p = 20) :
  (p + q) / 2 = 10 := by
  sorry

end arithmetic_mean_of_pq_is_10_l38_38497


namespace digit_place_value_ratio_l38_38127

theorem digit_place_value_ratio (n : ℚ) (h1 : n = 85247.2048) (h2 : ∃ d1 : ℚ, d1 * 0.1 = 0.2) (h3 : ∃ d2 : ℚ, d2 * 0.001 = 0.004) : 
  100 = 0.1 / 0.001 :=
by
  sorry

end digit_place_value_ratio_l38_38127


namespace tangent_line_through_point_l38_38412

theorem tangent_line_through_point (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) : 
  (∃ k : ℝ, 15 * x - 8 * y - 13 = 0) ∨ x = 3 := sorry

end tangent_line_through_point_l38_38412


namespace arithmetic_progression_terms_even_l38_38804

variable (a d : ℝ) (n : ℕ)

open Real

theorem arithmetic_progression_terms_even {n : ℕ} (hn_even : n % 2 = 0)
  (h_sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 32)
  (h_sum_even : (n / 2 : ℝ) * (2 * a + 2 * d + (n - 2) * d) = 40)
  (h_last_exceeds_first : (a + (n - 1) * d) - a = 8) : n = 16 :=
sorry

end arithmetic_progression_terms_even_l38_38804


namespace range_of_a_l38_38202

-- Define the function f
def f (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by {
  sorry
}

end range_of_a_l38_38202


namespace least_number_of_tiles_l38_38182

/-- A room of 544 cm long and 374 cm broad is to be paved with square tiles. 
    Prove that the least number of square tiles required to cover the floor is 176. -/
theorem least_number_of_tiles (length breadth : ℕ) (h1 : length = 544) (h2 : breadth = 374) :
  let gcd_length_breadth := Nat.gcd length breadth
  let num_tiles_length := length / gcd_length_breadth
  let num_tiles_breadth := breadth / gcd_length_breadth
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end least_number_of_tiles_l38_38182


namespace intersection_complement_A_B_l38_38634

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_A_B : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_B_l38_38634


namespace inscribed_square_side_length_l38_38837

theorem inscribed_square_side_length (a h : ℝ) (ha_pos : 0 < a) (hh_pos : 0 < h) :
    ∃ x : ℝ, x = (h * a) / (a + h) :=
by
  -- Code here demonstrates the existence of x such that x = (h * a) / (a + h)
  sorry

end inscribed_square_side_length_l38_38837


namespace A_and_B_work_together_for_49_days_l38_38161

variable (A B : ℝ)
variable (d : ℝ)
variable (fraction_left : ℝ)

def work_rate_A := 1 / 15
def work_rate_B := 1 / 20
def combined_work_rate := work_rate_A + work_rate_B

def fraction_work_completed (d : ℝ) := combined_work_rate * d

theorem A_and_B_work_together_for_49_days
    (A : ℝ := 1 / 15)
    (B : ℝ := 1 / 20)
    (fraction_left : ℝ := 0.18333333333333335) :
    (d : ℝ) → (fraction_work_completed d = 1 - fraction_left) →
    d = 49 :=
by
  sorry

end A_and_B_work_together_for_49_days_l38_38161


namespace minimum_tan_theta_is_sqrt7_l38_38558

noncomputable def min_tan_theta (z : ℂ) : ℝ := (Complex.abs (Complex.im z) / Complex.abs (Complex.re z))

theorem minimum_tan_theta_is_sqrt7 {z : ℂ} 
  (hz_real : 0 ≤ Complex.re z)
  (hz_imag : 0 ≤ Complex.im z)
  (hz_condition : Complex.abs (z^2 + 2) ≤ Complex.abs z) :
  min_tan_theta z = Real.sqrt 7 := sorry

end minimum_tan_theta_is_sqrt7_l38_38558


namespace slope_parallel_l38_38835

theorem slope_parallel {x y : ℝ} (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = -1/2 ∧ ( ∀ (x1 x2 : ℝ), 3 * x1 - 6 * y = 15 → ∃ y1 : ℝ, y1 = m * x1) :=
by
  sorry

end slope_parallel_l38_38835


namespace buy_items_ways_l38_38731

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l38_38731


namespace k_values_l38_38692

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def find_k (k : ℝ) : Prop :=
  (vector_dot (2, 3) (1, k) = 0) ∨
  (vector_dot (2, 3) (-1, k - 3) = 0) ∨
  (vector_dot (1, k) (-1, k - 3) = 0)

theorem k_values :
  ∃ k : ℝ, find_k k ∧ 
  (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end k_values_l38_38692


namespace total_legs_is_26_l38_38774

-- Define the number of puppies and chicks
def number_of_puppies : Nat := 3
def number_of_chicks : Nat := 7

-- Define the number of legs per puppy and per chick
def legs_per_puppy : Nat := 4
def legs_per_chick : Nat := 2

-- Calculate the total number of legs
def total_legs := (number_of_puppies * legs_per_puppy) + (number_of_chicks * legs_per_chick)

-- Prove that the total number of legs is 26
theorem total_legs_is_26 : total_legs = 26 := by
  sorry

end total_legs_is_26_l38_38774


namespace john_new_bench_press_l38_38985

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end john_new_bench_press_l38_38985


namespace trajectory_of_center_l38_38237

-- Define the fixed circle C as x^2 + (y + 3)^2 = 1
def fixed_circle (p : ℝ × ℝ) : Prop :=
  (p.1)^2 + (p.2 + 3)^2 = 1

-- Define the line y = 2
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2

-- The main theorem stating the trajectory of the center of circle M is x^2 = -12y
theorem trajectory_of_center :
  ∀ (M : ℝ × ℝ), 
  tangent_line M → (∃ r : ℝ, fixed_circle (M.1, M.2 - r) ∧ r > 0) →
  (M.1)^2 = -12 * M.2 :=
sorry

end trajectory_of_center_l38_38237


namespace man_speed_l38_38701

theorem man_speed (rest_time_per_km : ℕ := 5) (total_km_covered : ℕ := 5) (total_time_min : ℕ := 50) : 
  (total_time_min - rest_time_per_km * (total_km_covered - 1)) / 60 * total_km_covered = 10 := by
  sorry

end man_speed_l38_38701


namespace ratio_of_engineers_to_designers_l38_38164

-- Definitions of the variables
variables (e d : ℕ)

-- Conditions:
-- 1. The average age of the group is 45
-- 2. The average age of engineers is 40
-- 3. The average age of designers is 55

theorem ratio_of_engineers_to_designers (h : (40 * e + 55 * d) / (e + d) = 45) : e / d = 2 :=
by
-- Placeholder for the proof
sorry

end ratio_of_engineers_to_designers_l38_38164


namespace multiplication_problem_l38_38914

-- Definitions for different digits A, B, C, D
def is_digit (n : ℕ) := n < 10

theorem multiplication_problem 
  (A B C D : ℕ) 
  (hA : is_digit A) 
  (hB : is_digit B) 
  (hC : is_digit C) 
  (hD : is_digit D) 
  (h_diff : ∀ x y : ℕ, x ≠ y → is_digit x → is_digit y → x ≠ A → y ≠ B → x ≠ C → y ≠ D)
  (hD1 : D = 1)
  (h_mult : A * D = A) 
  (hC_eq : C = A + B) :
  A + C = 5 := sorry

end multiplication_problem_l38_38914


namespace video_duration_correct_l38_38228

/-
Define the conditions as given:
1. Vasya's time from home to school
2. Petya's time from school to home
3. Meeting conditions
-/

-- Define the times for Vasya and Petya
def vasya_time : ℕ := 8
def petya_time : ℕ := 5

-- Define the total video duration when correctly merged
def video_duration : ℕ := 5

-- State the theorem to be proved in Lean:
theorem video_duration_correct : vasya_time = 8 → petya_time = 5 → video_duration = 5 :=
by
  intros h1 h2
  exact sorry

end video_duration_correct_l38_38228


namespace price_of_each_tomato_l38_38616

theorem price_of_each_tomato
  (customers_per_month : ℕ)
  (lettuce_per_customer : ℕ)
  (lettuce_price : ℕ)
  (tomatoes_per_customer : ℕ)
  (total_monthly_sales : ℕ)
  (total_lettuce_sales : ℕ)
  (total_tomato_sales : ℕ)
  (price_per_tomato : ℝ)
  (h1 : customers_per_month = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomatoes_per_customer = 4)
  (h5 : total_monthly_sales = 2000)
  (h6 : total_lettuce_sales = customers_per_month * lettuce_per_customer * lettuce_price)
  (h7 : total_tomato_sales = total_monthly_sales - total_lettuce_sales)
  (h8 : total_lettuce_sales = 1000)
  (h9 : total_tomato_sales = 1000)
  (total_tomatoes_sold : ℕ := customers_per_month * tomatoes_per_customer)
  (h10 : total_tomatoes_sold = 2000) :
  price_per_tomato = 0.50 :=
by
  sorry

end price_of_each_tomato_l38_38616


namespace exists_xy_l38_38270

-- Given conditions from the problem
variables (m x0 y0 : ℕ)
-- Integers x0 and y0 are relatively prime
variables (rel_prim : Nat.gcd x0 y0 = 1)
-- y0 divides x0^2 + m
variables (div_y0 : y0 ∣ x0^2 + m)
-- x0 divides y0^2 + m
variables (div_x0 : x0 ∣ y0^2 + m)

-- Main theorem statement
theorem exists_xy 
  (hm : m > 0) 
  (hx0 : x0 > 0) 
  (hy0 : y0 > 0) 
  (rel_prim : Nat.gcd x0 y0 = 1) 
  (div_y0 : y0 ∣ x0^2 + m) 
  (div_x0 : x0 ∣ y0^2 + m) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ y ∣ x^2 + m ∧ x ∣ y^2 + m ∧ x + y ≤ m + 1 := 
sorry

end exists_xy_l38_38270


namespace points_per_right_answer_l38_38083

variable (p : ℕ)
variable (total_problems : ℕ := 25)
variable (wrong_problems : ℕ := 3)
variable (score : ℤ := 85)

theorem points_per_right_answer :
  (total_problems - wrong_problems) * p - wrong_problems = score -> p = 4 :=
  sorry

end points_per_right_answer_l38_38083


namespace businessman_expenditure_l38_38230

theorem businessman_expenditure (P : ℝ) (h1 : P * 1.21 = 24200) : P = 20000 := 
by sorry

end businessman_expenditure_l38_38230


namespace P_investment_calculation_l38_38491

variable {P_investment : ℝ}
variable (Q_investment : ℝ := 36000)
variable (total_profit : ℝ := 18000)
variable (Q_profit : ℝ := 6001.89)

def P_profit : ℝ := total_profit - Q_profit

theorem P_investment_calculation :
  P_investment = (P_profit * Q_investment) / Q_profit :=
by
  sorry

end P_investment_calculation_l38_38491


namespace complex_number_condition_l38_38401

theorem complex_number_condition (b : ℝ) :
  (2 + b) / 5 = (2 * b - 1) / 5 → b = 3 :=
by
  sorry

end complex_number_condition_l38_38401


namespace equation_has_seven_real_solutions_l38_38893

def f (x : ℝ) : ℝ := abs (x^2 - 1) - 1

theorem equation_has_seven_real_solutions (b c : ℝ) : 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) ↔ 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
  x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
  x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
  x₆ ≠ x₇ ∧
  f x₁ ^ 2 - b * f x₁ + c = 0 ∧ f x₂ ^ 2 - b * f x₂ + c = 0 ∧
  f x₃ ^ 2 - b * f x₃ + c = 0 ∧ f x₄ ^ 2 - b * f x₄ + c = 0 ∧
  f x₅ ^ 2 - b * f x₅ + c = 0 ∧ f x₆ ^ 2 - b * f x₆ + c = 0 ∧
  f x₇ ^ 2 - b * f x₇ + c = 0 :=
sorry

end equation_has_seven_real_solutions_l38_38893


namespace weak_multiple_l38_38421

def is_weak (a b n : ℕ) : Prop :=
  ∀ (x y : ℕ), n ≠ a * x + b * y

theorem weak_multiple (a b n : ℕ) (h_coprime : Nat.gcd a b = 1) (h_weak : is_weak a b n) (h_bound : n < a * b / 6) : 
  ∃ k ≥ 2, is_weak a b (k * n) :=
by
  sorry

end weak_multiple_l38_38421


namespace volume_of_wedge_l38_38604

theorem volume_of_wedge (h : 2 * Real.pi * r = 18 * Real.pi) :
  let V := (4 / 3) * Real.pi * (r ^ 3)
  let V_wedge := V / 6
  V_wedge = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l38_38604


namespace numbers_distance_one_neg_two_l38_38966

theorem numbers_distance_one_neg_two (x : ℝ) (h : abs (x + 2) = 1) : x = -1 ∨ x = -3 := 
sorry

end numbers_distance_one_neg_two_l38_38966


namespace midpoint_in_polar_coordinates_l38_38953

-- Define the problem as a theorem in Lean 4
theorem midpoint_in_polar_coordinates :
  let A := (10, Real.pi / 4)
  let B := (10, 3 * Real.pi / 4)
  ∃ r θ, (r = 5 * Real.sqrt 2) ∧ (θ = Real.pi / 2) ∧
         0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

end midpoint_in_polar_coordinates_l38_38953


namespace polar_to_cartesian_l38_38666

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.cos θ) :
  ∃ x y : ℝ, (x=ρ*Real.cos θ ∧ y=ρ*Real.sin θ) ∧ (x-1)^2 + y^2 = 1 :=
by
  sorry

end polar_to_cartesian_l38_38666


namespace square_area_eq_36_l38_38218

theorem square_area_eq_36 :
  let triangle_side1 := 5.5
  let triangle_side2 := 7.5
  let triangle_side3 := 11
  let triangle_perimeter := triangle_side1 + triangle_side2 + triangle_side3
  let square_perimeter := triangle_perimeter
  let square_side_length := square_perimeter / 4
  let square_area := square_side_length * square_side_length
  square_area = 36 := by
  sorry

end square_area_eq_36_l38_38218


namespace square_of_ratio_is_specified_value_l38_38528

theorem square_of_ratio_is_specified_value (a b c : ℝ) (h1 : c = Real.sqrt (a^2 + b^2)) (h2 : a / b = b / c) :
  (a / b)^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end square_of_ratio_is_specified_value_l38_38528


namespace find_minimum_value_l38_38162

-- This definition captures the condition that a, b, c are positive real numbers
def pos_reals := { x : ℝ // 0 < x }

-- The main theorem statement
theorem find_minimum_value (a b c : pos_reals) :
  4 * (a.1 ^ 4) + 8 * (b.1 ^ 4) + 16 * (c.1 ^ 4) + 1 / (a.1 * b.1 * c.1) ≥ 10 :=
by
  -- This is where the proof will go
  sorry

end find_minimum_value_l38_38162


namespace smallest_n_l38_38675

def power_tower (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => a
  | (n+1) => a ^ (power_tower a n)

def pow3_cubed : ℕ := 3 ^ (3 ^ (3 ^ 3))

theorem smallest_n : ∃ n, (∃ k : ℕ, (power_tower 2 n) = k ∧ k > pow3_cubed) ∧ ∀ m, (∃ k : ℕ, (power_tower 2 m) = k ∧ k > pow3_cubed) → m ≥ n :=
  by
  sorry

end smallest_n_l38_38675


namespace determinant_of_matrix4x5_2x3_l38_38132

def matrix4x5_2x3 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 5], ![2, 3]]

theorem determinant_of_matrix4x5_2x3 : matrix4x5_2x3.det = 2 := 
by
  sorry

end determinant_of_matrix4x5_2x3_l38_38132


namespace cubic_sum_expression_l38_38251

theorem cubic_sum_expression (x y z p q r : ℝ) (h1 : x * y = p) (h2 : x * z = q) (h3 : y * z = r) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by
  sorry

end cubic_sum_expression_l38_38251


namespace combined_weight_difference_l38_38061

def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := -5.25
def biology_weight : ℝ := 3.755

theorem combined_weight_difference :
  (chemistry_weight - calculus_weight) - (geometry_weight + biology_weight) = 7.995 :=
by
  sorry

end combined_weight_difference_l38_38061


namespace total_fruits_l38_38534

theorem total_fruits (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 4) : a + b + c = 15 := by
  sorry

end total_fruits_l38_38534


namespace sum_a5_a8_l38_38098

variable (a : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_a5_a8 (a1 a2 a3 a4 : ℝ) (q : ℝ)
  (h1 : a1 + a3 = 1)
  (h2 : a2 + a4 = 2)
  (h_seq : is_geometric_sequence a q)
  (a_def : ∀ n : ℕ, a n = a1 * q^n) :
  a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end sum_a5_a8_l38_38098


namespace max_at_zero_l38_38972

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_at_zero : ∀ x : ℝ, f x ≤ f 0 :=
by
  sorry

end max_at_zero_l38_38972


namespace heartsuit_ratio_l38_38879

-- Define the operation ⧡
def heartsuit (n m : ℕ) := n^(3+m) * m^(2+n)

-- The problem statement to prove
theorem heartsuit_ratio : heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l38_38879


namespace player_B_wins_l38_38996

-- Here we define the scenario and properties from the problem statement.
def initial_pile1 := 100
def initial_pile2 := 252

-- Definition of a turn, conditions and the win condition based on the problem
structure Turn :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (player_A_turn : Bool)  -- True if it's player A's turn, False if it's player B's turn

-- The game conditions and strategy for determining the winner
def will_player_B_win (initial_pile1 initial_pile2 : ℕ) : Bool :=
  -- assuming the conditions are provided and correctly analyzed, 
  -- we directly state the known result according to the optimal strategies from the solution
  true  -- B wins as per the solution's analysis if both play optimally.

-- The final theorem stating Player B wins given the initial conditions with both playing optimally and A going first.
theorem player_B_wins : will_player_B_win initial_pile1 initial_pile2 = true :=
  sorry  -- Proof omitted.

end player_B_wins_l38_38996


namespace original_grain_correct_l38_38794

-- Define the initial quantities
def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

-- Define the original amount of grain expected
def original_grain : ℕ := 50870

-- Prove that the original amount of grain was correct
theorem original_grain_correct : grain_spilled + grain_remaining = original_grain := 
by
  sorry

end original_grain_correct_l38_38794


namespace compare_inequalities_l38_38987

theorem compare_inequalities (a b c π : ℝ) (h1 : a > π) (h2 : π > b) (h3 : b > 1) (h4 : 1 > c) (h5 : c > 0) 
  (x := a^(1 / π)) (y := Real.log b / Real.log π) (z := Real.log π / Real.log c) : x > y ∧ y > z := 
sorry

end compare_inequalities_l38_38987


namespace longer_piece_length_is_20_l38_38754

-- Define the rope length
def ropeLength : ℕ := 35

-- Define the ratio of the two pieces
def ratioA : ℕ := 3
def ratioB : ℕ := 4
def totalRatio : ℕ := ratioA + ratioB

-- Define the length of each part
def partLength : ℕ := ropeLength / totalRatio

-- Define the length of the longer piece
def longerPieceLength : ℕ := ratioB * partLength

-- Theorem to prove that the length of the longer piece is 20 inches
theorem longer_piece_length_is_20 : longerPieceLength = 20 := by 
  sorry

end longer_piece_length_is_20_l38_38754


namespace passengers_got_on_in_Texas_l38_38453

theorem passengers_got_on_in_Texas (start_pax : ℕ) 
  (texas_depart_pax : ℕ) 
  (nc_depart_pax : ℕ) 
  (nc_board_pax : ℕ) 
  (virginia_total_people : ℕ) 
  (crew_members : ℕ) 
  (final_pax_virginia : ℕ) 
  (X : ℕ) :
  start_pax = 124 →
  texas_depart_pax = 58 →
  nc_depart_pax = 47 →
  nc_board_pax = 14 →
  virginia_total_people = 67 →
  crew_members = 10 →
  final_pax_virginia = virginia_total_people - crew_members →
  X + 33 = final_pax_virginia →
  X = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end passengers_got_on_in_Texas_l38_38453


namespace required_height_for_roller_coaster_l38_38027

-- Definitions based on conditions from the problem
def initial_height : ℕ := 48
def natural_growth_rate_per_month : ℚ := 1 / 3
def upside_down_growth_rate_per_hour : ℚ := 1 / 12
def hours_per_month_hanging_upside_down : ℕ := 2
def months_in_a_year : ℕ := 12

-- Calculations needed for the proof
def annual_natural_growth := natural_growth_rate_per_month * months_in_a_year
def annual_upside_down_growth := (upside_down_growth_rate_per_hour * hours_per_month_hanging_upside_down) * months_in_a_year
def total_annual_growth := annual_natural_growth + annual_upside_down_growth
def height_next_year := initial_height + total_annual_growth

-- Statement of the required height for the roller coaster
theorem required_height_for_roller_coaster : height_next_year = 54 :=
by
  sorry

end required_height_for_roller_coaster_l38_38027


namespace unit_digit_23_pow_100000_l38_38226

theorem unit_digit_23_pow_100000 : (23^100000) % 10 = 1 := 
by
  -- Import necessary submodules and definitions

sorry

end unit_digit_23_pow_100000_l38_38226


namespace numerator_of_fraction_l38_38223

/-- 
Given:
1. The denominator of a fraction is 7 less than 3 times the numerator.
2. The fraction is equivalent to 2/5.
Prove that the numerator of the fraction is 14.
-/
theorem numerator_of_fraction {x : ℕ} (h : x / (3 * x - 7) = 2 / 5) : x = 14 :=
  sorry

end numerator_of_fraction_l38_38223


namespace intersection_A_B_l38_38764

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 4} :=
sorry

end intersection_A_B_l38_38764


namespace distinct_real_roots_find_k_values_l38_38641

-- Question 1: Prove the equation has two distinct real roots
theorem distinct_real_roots (k : ℝ) : 
  (2 * k + 1) ^ 2 - 4 * (k ^ 2 + k) > 0 :=
  by sorry

-- Question 2: Find the values of k when triangle ABC is a right triangle
theorem find_k_values (k : ℝ) : 
  (k = 3 ∨ k = 12) ↔ 
  (∃ (AB AC : ℝ), 
    AB ≠ AC ∧ AB = k ∧ AC = k + 1 ∧ (AB^2 + AC^2 = 5^2 ∨ AC^2 + 5^2 = AB^2)) :=
  by sorry

end distinct_real_roots_find_k_values_l38_38641


namespace smallest_five_digit_divisible_by_53_l38_38121

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l38_38121


namespace least_positive_integer_div_conditions_l38_38609

theorem least_positive_integer_div_conditions :
  ∃ n > 1, (n % 4 = 3) ∧ (n % 5 = 3) ∧ (n % 7 = 3) ∧ (n % 10 = 3) ∧ (n % 11 = 3) ∧ n = 1543 := 
by 
  sorry

end least_positive_integer_div_conditions_l38_38609


namespace sum_of_roots_of_equation_l38_38580

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l38_38580


namespace james_oranges_l38_38071

-- Define the problem conditions
variables (o a : ℕ) -- o is number of oranges, a is number of apples

-- Condition: James bought apples and oranges over a seven-day week
def days_week := o + a = 7

-- Condition: The total cost must be a whole number of dollars (divisible by 100 cents)
def total_cost := 65 * o + 40 * a ≡ 0 [MOD 100]

-- We need to prove: James bought 4 oranges
theorem james_oranges (o a : ℕ) (h_days_week : days_week o a) (h_total_cost : total_cost o a) : o = 4 :=
sorry

end james_oranges_l38_38071


namespace calc_expression_correct_l38_38370

noncomputable def calc_expression : Real :=
  Real.sqrt 8 - (1 / 3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2

theorem calc_expression_correct :
  calc_expression = 3 - Real.sqrt 3 :=
sorry

end calc_expression_correct_l38_38370


namespace area_of_black_parts_l38_38110

theorem area_of_black_parts (x y : ℕ) (h₁ : x + y = 106) (h₂ : x + 2 * y = 170) : y = 64 :=
sorry

end area_of_black_parts_l38_38110


namespace cylinder_radius_inscribed_box_l38_38277

theorem cylinder_radius_inscribed_box :
  ∀ (x y z r : ℝ),
    4 * (x + y + z) = 160 →
    2 * (x * y + y * z + x * z) = 600 →
    z = 40 - x - y →
    r = (1/2) * Real.sqrt (x^2 + y^2) →
    r = (15 * Real.sqrt 2) / 2 :=
by
  sorry

end cylinder_radius_inscribed_box_l38_38277


namespace fixed_point_on_line_l38_38214

theorem fixed_point_on_line (m x y : ℝ) (h : ∀ m : ℝ, m * x - y + 2 * m + 1 = 0) : 
  (x = -2 ∧ y = 1) :=
sorry

end fixed_point_on_line_l38_38214


namespace remainder_of_power_mod_l38_38305

theorem remainder_of_power_mod (a n p : ℕ) (h_prime : Nat.Prime p) (h_a : a < p) :
  (3 : ℕ)^2024 % 17 = 13 :=
by
  sorry

end remainder_of_power_mod_l38_38305


namespace gcd_condition_l38_38725

def seq (a : ℕ → ℕ) := a 0 = 3 ∧ ∀ n, a (n + 1) - a n = n * (a n - 1)

theorem gcd_condition (a : ℕ → ℕ) (m : ℕ) (h : seq a) :
  m ≥ 2 → (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 := 
sorry

end gcd_condition_l38_38725


namespace sum_of_center_coordinates_l38_38526

def center_of_circle_sum (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 101

theorem sum_of_center_coordinates : center_of_circle_sum x y → x + y = 1 :=
sorry

end sum_of_center_coordinates_l38_38526


namespace prop_A_prop_B_prop_C_prop_D_l38_38849

variable {a b : ℝ}

-- Proposition A
theorem prop_A (h : a^2 - b^2 = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b < 1 := sorry

-- Proposition B (negation of the original proposition since B is incorrect)
theorem prop_B (h : (1 / b) - (1 / a) = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b ≥ 1 := sorry

-- Proposition C
theorem prop_C (h : a > b + 1) (a_pos : 0 < a) (b_pos : 0 < b) : a^2 > b^2 + 1 := sorry

-- Proposition D (negation of the original proposition since D is incorrect)
theorem prop_D (h1 : a ≤ 1) (h2 : b ≤ 1) (a_pos : 0 < a) (b_pos : 0 < b) : |a - b| < |1 - a * b| := sorry

end prop_A_prop_B_prop_C_prop_D_l38_38849


namespace speed_of_stream_l38_38697

theorem speed_of_stream (v c : ℝ) (h1 : c - v = 6) (h2 : c + v = 10) : v = 2 :=
by
  sorry

end speed_of_stream_l38_38697


namespace fill_trough_time_l38_38107

noncomputable def time_to_fill (T_old T_new T_third : ℕ) : ℝ :=
  let rate_old := (1 : ℝ) / T_old
  let rate_new := (1 : ℝ) / T_new
  let rate_third := (1 : ℝ) / T_third
  let total_rate := rate_old + rate_new + rate_third
  1 / total_rate

theorem fill_trough_time:
  time_to_fill 600 200 400 = 1200 / 11 := 
by
  sorry

end fill_trough_time_l38_38107


namespace compute_expr_l38_38433

theorem compute_expr :
  ((π - 3.14)^0 + (-0.125)^2008 * 8^2008) = 2 := 
by 
  sorry

end compute_expr_l38_38433


namespace phi_cannot_be_chosen_l38_38623

theorem phi_cannot_be_chosen (θ φ : ℝ) (hθ : -π/2 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π)
  (h1 : 3 * Real.sin θ = 3 * Real.sqrt 2 / 2) 
  (h2 : 3 * Real.sin (-2*φ + θ) = 3 * Real.sqrt 2 / 2) : φ ≠ 5*π/4 :=
by
  sorry

end phi_cannot_be_chosen_l38_38623


namespace custom_op_neg2_neg3_l38_38456

  def custom_op (a b : ℤ) : ℤ := b^2 - a

  theorem custom_op_neg2_neg3 : custom_op (-2) (-3) = 11 :=
  by
    sorry
  
end custom_op_neg2_neg3_l38_38456


namespace min_possible_range_l38_38665

theorem min_possible_range (A B C : ℤ) : 
  (A + 15 ≤ C ∧ B + 25 ≤ C ∧ C ≤ A + 45) → C - A ≤ 45 :=
by
  intros h
  have h1 : A + 15 ≤ C := h.1
  have h2 : B + 25 ≤ C := h.2.1
  have h3 : C ≤ A + 45 := h.2.2
  sorry

end min_possible_range_l38_38665


namespace cuboid_third_edge_l38_38208

theorem cuboid_third_edge (a b V h : ℝ) (ha : a = 4) (hb : b = 4) (hV : V = 96) (volume_formula : V = a * b * h) : h = 6 :=
by
  sorry

end cuboid_third_edge_l38_38208


namespace solve_problem_l38_38432

open Nat

theorem solve_problem :
  ∃ (n p : ℕ), p.Prime ∧ n > 0 ∧ ∃ k : ℤ, p^2 + 7^n = k^2 ∧ (n, p) = (1, 3) := 
by
  sorry

end solve_problem_l38_38432


namespace Aiyanna_cookies_l38_38546

-- Define the conditions
def Alyssa_cookies : ℕ := 129
variable (x : ℕ)
def difference_condition : Prop := (Alyssa_cookies - x) = 11

-- The theorem to prove
theorem Aiyanna_cookies (x : ℕ) (h : difference_condition x) : x = 118 :=
by sorry

end Aiyanna_cookies_l38_38546


namespace max_value_m_l38_38656

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (m : ℝ)
  (h : (2 / a) + (1 / b) ≥ m / (2 * a + b)) : m ≤ 9 :=
sorry

end max_value_m_l38_38656


namespace price_reduction_l38_38913

theorem price_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 150 * (1 - x) * (1 - x) = 96 :=
sorry

end price_reduction_l38_38913


namespace max_brownies_l38_38741

-- Definitions for the conditions given in the problem
def is_interior_pieces (m n : ℕ) : ℕ := (m - 2) * (n - 2)
def is_perimeter_pieces (m n : ℕ) : ℕ := 2 * m + 2 * n - 4

-- The assertion that the number of brownies along the perimeter is twice the number in the interior
def condition (m n : ℕ) : Prop := 2 * is_interior_pieces m n = is_perimeter_pieces m n

-- The statement that the maximum number of brownies under the given condition is 84
theorem max_brownies : ∃ (m n : ℕ), condition m n ∧ m * n = 84 := by
  sorry

end max_brownies_l38_38741


namespace expected_steps_unit_interval_l38_38909

noncomputable def expected_steps_to_color_interval : ℝ := 
  -- Placeholder for the function calculating expected steps
  sorry 

theorem expected_steps_unit_interval : expected_steps_to_color_interval = 5 :=
  sorry

end expected_steps_unit_interval_l38_38909


namespace sum_after_operations_l38_38922

theorem sum_after_operations (a b S : ℝ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := 
by 
  sorry

end sum_after_operations_l38_38922


namespace students_contribution_l38_38019

theorem students_contribution (n x : ℕ) 
  (h₁ : ∃ (k : ℕ), k * 9 = 22725)
  (h₂ : n * x = k / 9)
  : (n = 5 ∧ x = 505) ∨ (n = 25 ∧ x = 101) :=
sorry

end students_contribution_l38_38019


namespace profit_share_difference_l38_38059

theorem profit_share_difference (P : ℝ) (hP : P = 1000) 
  (rX rY : ℝ) (hRatio : rX / rY = (1/2) / (1/3)) : 
  let total_parts := (1/2) + (1/3)
  let value_per_part := P / total_parts
  let x_share := (1/2) * value_per_part
  let y_share := (1/3) * value_per_part
  x_share - y_share = 200 := by 
  sorry

end profit_share_difference_l38_38059


namespace charlie_has_32_cards_l38_38810

variable (Chris_cards Charlie_cards : ℕ)

def chris_has_18_cards : Chris_cards = 18 := sorry
def chris_has_14_fewer_cards_than_charlie : Chris_cards + 14 = Charlie_cards := sorry

theorem charlie_has_32_cards (h18 : Chris_cards = 18) (h14 : Chris_cards + 14 = Charlie_cards) : Charlie_cards = 32 := 
sorry

end charlie_has_32_cards_l38_38810


namespace smallest_three_digit_multiple_of_17_l38_38852

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l38_38852


namespace find_number_l38_38117

theorem find_number (x : ℝ) (h : (1/2) * x + 7 = 17) : x = 20 :=
sorry

end find_number_l38_38117


namespace coloring_ways_of_circle_l38_38267

noncomputable def num_ways_to_color_circle (n : ℕ) (k : ℕ) : ℕ :=
  if h : n % 2 = 1 then -- There are 13 parts; n must be odd (since adjacent matching impossible in even n)
    (k * (k - 1)^(n - 1) : ℕ)
  else
    0

theorem coloring_ways_of_circle :
  num_ways_to_color_circle 13 3 = 6 :=
by
  sorry

end coloring_ways_of_circle_l38_38267


namespace total_books_after_donations_l38_38871

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end total_books_after_donations_l38_38871


namespace darren_and_fergie_same_amount_in_days_l38_38542

theorem darren_and_fergie_same_amount_in_days : 
  ∀ (t : ℕ), (200 + 16 * t = 300 + 12 * t) → t = 25 := 
by sorry

end darren_and_fergie_same_amount_in_days_l38_38542


namespace swimming_speed_in_still_water_l38_38366

theorem swimming_speed_in_still_water 
  (speed_of_water : ℝ) (distance : ℝ) (time : ℝ) (v : ℝ) 
  (h_water_speed : speed_of_water = 2) 
  (h_time_distance : time = 4 ∧ distance = 8) :
  v = 4 :=
by
  sorry

end swimming_speed_in_still_water_l38_38366


namespace minimize_on_interval_l38_38102

def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

theorem minimize_on_interval (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≥ if a < 0 then -2 else if 0 ≤ a ∧ a ≤ 2 then -a^2 - 2 else 2 - 4*a) :=
by 
  sorry

end minimize_on_interval_l38_38102


namespace union_of_A_and_B_l38_38313

open Set

variable {α : Type}

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := 
by
  sorry

end union_of_A_and_B_l38_38313


namespace axis_of_symmetry_condition_l38_38861

theorem axis_of_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
    (h_sym : ∀ x y, y = -x → y = (p * x + q) / (r * x + s)) : p = s :=
by
  sorry

end axis_of_symmetry_condition_l38_38861


namespace find_ratio_l38_38590

def celsius_to_fahrenheit_ratio (ratio : ℝ) (c f : ℝ) : Prop :=
  f = ratio * c + 32

theorem find_ratio (ratio : ℝ) :
  (∀ c f, celsius_to_fahrenheit_ratio ratio c f ∧ ((f = 58) → (c = 14.444444444444445)) → f = 1.8 * c + 32) ∧ 
  (f - 32 = ratio * (c - 0)) ∧
  (c = 14.444444444444445 → f = 32 + 26) ∧
  (f = 58 → c = 14.444444444444445) ∧ 
  (ratio = 1.8)
  → ratio = 1.8 := 
sorry 


end find_ratio_l38_38590


namespace find_LCM_of_numbers_l38_38114

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers_l38_38114


namespace trip_distance_1200_miles_l38_38888

theorem trip_distance_1200_miles
    (D : ℕ)
    (H : D / 50 - D / 60 = 4) :
    D = 1200 :=
by
    sorry

end trip_distance_1200_miles_l38_38888


namespace inequality_am_gm_l38_38193

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l38_38193


namespace maryville_population_increase_l38_38484

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l38_38484


namespace relationship_of_x_vals_l38_38690

variables {k x1 x2 x3 : ℝ}

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem relationship_of_x_vals (h1 : inverse_proportion_function k x1 = 1)
                              (h2 : inverse_proportion_function k x2 = -5)
                              (h3 : inverse_proportion_function k x3 = 3)
                              (hk : k < 0) :
                              x1 < x3 ∧ x3 < x2 :=
by
  sorry

end relationship_of_x_vals_l38_38690


namespace max_ants_collisions_l38_38296

theorem max_ants_collisions (n : ℕ) (hpos : 0 < n) :
  ∃ (ants : Fin n → ℝ) (speeds: Fin n → ℝ) (finite_collisions : Prop)
    (collisions_bound : ℕ),
  (∀ i : Fin n, speeds i ≠ 0) →
  finite_collisions →
  collisions_bound = (n * (n - 1)) / 2 :=
by
  sorry

end max_ants_collisions_l38_38296


namespace sum_of_first_50_primes_is_5356_l38_38411

open Nat

-- Define the first 50 prime numbers
def first_50_primes : List Nat := 
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
   83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

-- Calculate their sum
def sum_first_50_primes : Nat := List.foldr (Nat.add) 0 first_50_primes

-- Now we state the theorem we want to prove
theorem sum_of_first_50_primes_is_5356 : 
  sum_first_50_primes = 5356 := 
by
  -- Placeholder for proof
  sorry

end sum_of_first_50_primes_is_5356_l38_38411


namespace calories_per_candy_bar_l38_38921

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) 
  (h : total_calories = 341) (n : number_of_bars = 11) : (total_calories / number_of_bars = 31) :=
by
  sorry

end calories_per_candy_bar_l38_38921


namespace log_a_properties_l38_38840

noncomputable def log_a (a x : ℝ) (h : 0 < a ∧ a < 1) : ℝ := Real.log x / Real.log a

theorem log_a_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 1 < x → log_a a x h < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → log_a a x h > 0) ∧
  (¬ ∀ x1 x2 : ℝ, log_a a x1 h > log_a a x2 h → x1 > x2) ∧
  (∀ x y : ℝ, log_a a (x * y) h = log_a a x h + log_a a y h) :=
by
  sorry

end log_a_properties_l38_38840


namespace sum_of_x_coordinates_mod13_intersection_l38_38591

theorem sum_of_x_coordinates_mod13_intersection :
  (∀ x y : ℕ, y ≡ 3 * x + 5 [MOD 13] → y ≡ 7 * x + 4 [MOD 13]) → (x ≡ 10 [MOD 13]) :=
by
  sorry

end sum_of_x_coordinates_mod13_intersection_l38_38591


namespace find_value_of_6b_l38_38857

theorem find_value_of_6b (a b : ℝ) (h1 : 10 * a = 20) (h2 : 120 * a * b = 800) : 6 * b = 20 :=
by
  sorry

end find_value_of_6b_l38_38857


namespace min_value_one_div_a_plus_one_div_b_l38_38645

theorem min_value_one_div_a_plus_one_div_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_one_div_a_plus_one_div_b_l38_38645


namespace math_problem_l38_38851

variable (a b c : ℝ)

variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a ≠ -b) (h5 : b ≠ -c) (h6 : c ≠ -a)

theorem math_problem 
    (h₁ : (a * b) / (a + b) = 4)
    (h₂ : (b * c) / (b + c) = 5)
    (h₃ : (c * a) / (c + a) = 7) :
    (a * b * c) / (a * b + b * c + c * a) = 280 / 83 := 
sorry

end math_problem_l38_38851


namespace find_f_neg_a_l38_38954

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l38_38954


namespace expand_product_eq_l38_38676

theorem expand_product_eq :
  (∀ (x : ℤ), (x^3 - 3 * x^2 + 3 * x - 1) * (x^2 + 3 * x + 3) = x^5 - 3 * x^3 - x^2 + 3 * x) :=
by
  intro x
  sorry

end expand_product_eq_l38_38676


namespace f_properties_l38_38637

noncomputable def f (x : ℝ) : ℝ :=
if -2 < x ∧ x < 0 then 2^x else sorry

theorem f_properties (f_odd : ∀ x : ℝ, f (-x) = -f x)
                     (f_periodic : ∀ x : ℝ, f (x + 3 / 2) = -f x) :
  f 2014 + f 2015 + f 2016 = 0 :=
by 
  -- The proof will go here
  sorry

end f_properties_l38_38637


namespace bud_age_is_eight_l38_38723

def uncle_age : ℕ := 24

def bud_age (uncle_age : ℕ) : ℕ := uncle_age / 3

theorem bud_age_is_eight : bud_age uncle_age = 8 :=
by
  sorry

end bud_age_is_eight_l38_38723


namespace min_value_expression_l38_38763

theorem min_value_expression 
  (a b c : ℝ)
  (h1 : a + b + c = -1)
  (h2 : a * b * c ≤ -3) : 
  (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) ≥ 3 :=
sorry

end min_value_expression_l38_38763


namespace statues_ratio_l38_38150

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio_l38_38150


namespace prove_non_negative_axbycz_l38_38618

variable {a b c x y z : ℝ}

theorem prove_non_negative_axbycz
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := 
sorry

end prove_non_negative_axbycz_l38_38618


namespace factorial_expression_evaluation_l38_38897

theorem factorial_expression_evaluation : (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 2))^2 = 1440 :=
by
  sorry

end factorial_expression_evaluation_l38_38897


namespace average_weight_of_remaining_students_l38_38776

theorem average_weight_of_remaining_students
  (M F M' F' : ℝ) (A A' : ℝ)
  (h1 : M + F = 60 * A)
  (h2 : M' + F' = 59 * A')
  (h3 : A' = A + 0.2)
  (h4 : M' = M - 45):
  A' = 57 :=
by
  sorry

end average_weight_of_remaining_students_l38_38776


namespace probability_of_rolling_five_l38_38332

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end probability_of_rolling_five_l38_38332


namespace necessary_but_not_sufficient_l38_38504

theorem necessary_but_not_sufficient (a b : ℝ) : 
 (a > b) ↔ (a-1 > b+1) :=
by {
  sorry
}

end necessary_but_not_sufficient_l38_38504


namespace digits_with_five_or_seven_is_5416_l38_38054

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l38_38054


namespace absent_present_probability_l38_38759

theorem absent_present_probability : 
  ∀ (p_absent_normal p_absent_workshop p_present_workshop : ℚ), 
    p_absent_normal = 1 / 20 →
    p_absent_workshop = 2 * p_absent_normal →
    p_present_workshop = 1 - p_absent_workshop →
    p_absent_workshop = 1 / 10 →
    (p_present_workshop * p_absent_workshop + p_absent_workshop * p_present_workshop) * 100 = 18 :=
by
  intros
  sorry

end absent_present_probability_l38_38759


namespace time_to_cover_escalator_l38_38039

noncomputable def escalator_speed : ℝ := 8
noncomputable def person_speed : ℝ := 2
noncomputable def escalator_length : ℝ := 160
noncomputable def combined_speed : ℝ := escalator_speed + person_speed

theorem time_to_cover_escalator :
  escalator_length / combined_speed = 16 := by
  sorry

end time_to_cover_escalator_l38_38039


namespace seating_arrangement_l38_38065

theorem seating_arrangement (x y : ℕ) (h1 : 9 * x + 7 * y = 61) : x = 6 :=
by 
  sorry

end seating_arrangement_l38_38065


namespace Shara_borrowed_6_months_ago_l38_38640

theorem Shara_borrowed_6_months_ago (X : ℝ) (h1 : ∃ n : ℕ, (X / 2 - 4 * 10 = 20) ∧ (X / 2 = n * 10)) :
  ∃ m : ℕ, m * 10 = X / 2 → m = 6 := 
sorry

end Shara_borrowed_6_months_ago_l38_38640


namespace tangent_circle_exists_l38_38082
open Set

-- Definitions of given point, line, and circle
variables {Point : Type*} {Line : Type*} {Circle : Type*} 
variables (M : Point) (l : Line) (S : Circle)
variables (center_S : Point) (radius_S : ℝ)

-- Conditions of the problem
variables (touches_line : Circle → Line → Prop) (touches_circle : Circle → Circle → Prop)
variables (passes_through : Circle → Point → Prop) (center_of : Circle → Point)
variables (radius_of : Circle → ℝ)

-- Existence theorem to prove
theorem tangent_circle_exists 
  (given_tangent_to_line : Circle → Line → Bool)
  (given_tangent_to_circle : Circle → Circle → Bool)
  (given_passes_through : Circle → Point → Bool):
  ∃ (Ω : Circle), 
    given_tangent_to_line Ω l ∧
    given_tangent_to_circle Ω S ∧
    given_passes_through Ω M :=
sorry

end tangent_circle_exists_l38_38082


namespace arithmetic_sequence_a12_l38_38732

theorem arithmetic_sequence_a12 (a : ℕ → ℝ)
    (h1 : a 3 + a 4 + a 5 = 3)
    (h2 : a 8 = 8)
    (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) :
    a 12 = 15 :=
by
  -- Since we aim to ensure the statement alone compiles, we leave the proof with 'sorry'.
  sorry

end arithmetic_sequence_a12_l38_38732


namespace market_value_of_stock_l38_38845

-- Define the given conditions.
def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.09 * face_value
def yield : ℝ := 0.08

-- State the problem: proving the market value of the stock.
theorem market_value_of_stock : (dividend_per_share / yield) * 100 = 112.50 := by
  -- Placeholder for the proof
  sorry

end market_value_of_stock_l38_38845


namespace figure_square_count_l38_38067

theorem figure_square_count (f : ℕ → ℕ)
  (h0 : f 0 = 2)
  (h1 : f 1 = 8)
  (h2 : f 2 = 18)
  (h3 : f 3 = 32) :
  f 100 = 20402 :=
sorry

end figure_square_count_l38_38067


namespace non_zero_real_x_solution_l38_38547

theorem non_zero_real_x_solution (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 := by
  sorry

end non_zero_real_x_solution_l38_38547


namespace residue_calculation_l38_38529

theorem residue_calculation 
  (h1 : 182 ≡ 0 [MOD 14])
  (h2 : 182 * 12 ≡ 0 [MOD 14])
  (h3 : 15 * 7 ≡ 7 [MOD 14])
  (h4 : 3 ≡ 3 [MOD 14]) :
  (182 * 12 - 15 * 7 + 3) ≡ 10 [MOD 14] :=
sorry

end residue_calculation_l38_38529


namespace first_year_with_sum_of_digits_10_after_2020_l38_38463

theorem first_year_with_sum_of_digits_10_after_2020 :
  ∃ (y : ℕ), y > 2020 ∧ (y.digits 10).sum = 10 ∧ ∀ (z : ℕ), (z > 2020 ∧ (z.digits 10).sum = 10) → y ≤ z :=
sorry

end first_year_with_sum_of_digits_10_after_2020_l38_38463


namespace min_value_sequence_l38_38490

theorem min_value_sequence (a : ℕ → ℕ) (h1 : a 2 = 102) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) - a n = 4 * n) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (a m) / m ≥ 26) :=
sorry

end min_value_sequence_l38_38490


namespace mary_can_keep_warm_l38_38784

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l38_38784


namespace julia_drove_miles_l38_38668

theorem julia_drove_miles :
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ),
    daily_rental_cost = 29 →
    cost_per_mile = 0.08 →
    total_paid = 46.12 →
    total_paid - daily_rental_cost = cost_per_mile * 214 :=
by
  intros _ _ _ d_cost_eq cpm_eq tp_eq
  -- calculation and proof steps will be filled here
  sorry

end julia_drove_miles_l38_38668


namespace calculate_expression_l38_38739

theorem calculate_expression : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end calculate_expression_l38_38739


namespace probability_cs_majors_consecutive_l38_38466

def total_ways_to_choose_5_out_of_12 : ℕ :=
  Nat.choose 12 5

def number_of_ways_cs_majors_consecutive : ℕ :=
  12

theorem probability_cs_majors_consecutive :
  (number_of_ways_cs_majors_consecutive : ℚ) / (total_ways_to_choose_5_out_of_12 : ℚ) = 1 / 66 := by
  sorry

end probability_cs_majors_consecutive_l38_38466


namespace remainder_is_three_l38_38579

def dividend : ℕ := 15
def divisor : ℕ := 3
def quotient : ℕ := 4

theorem remainder_is_three : dividend = (divisor * quotient) + Nat.mod dividend divisor := by
  sorry

end remainder_is_three_l38_38579


namespace y_value_when_x_is_20_l38_38573

theorem y_value_when_x_is_20 :
  ∀ (x : ℝ), (∀ m c : ℝ, m = 2.5 → c = 3 → (y = m * x + c) → x = 20 → y = 53) :=
by
  sorry

end y_value_when_x_is_20_l38_38573


namespace division_by_fraction_l38_38012

theorem division_by_fraction :
  5 / (8 / 13) = 65 / 8 :=
sorry

end division_by_fraction_l38_38012


namespace remainder_of_3_pow_45_mod_17_l38_38010

theorem remainder_of_3_pow_45_mod_17 : 3^45 % 17 = 15 := 
by {
  sorry
}

end remainder_of_3_pow_45_mod_17_l38_38010


namespace double_people_half_work_l38_38823

-- Definitions
def initial_person_count (P : ℕ) : Prop := true
def initial_time (T : ℕ) : Prop := T = 16

-- Theorem
theorem double_people_half_work (P T : ℕ) (hP : initial_person_count P) (hT : initial_time T) : P > 0 → (2 * P) * (T / 2) = P * T / 2 := by
  sorry

end double_people_half_work_l38_38823


namespace incorrect_statement_D_l38_38751

theorem incorrect_statement_D :
  ¬ (abs (-1) - abs 1 = 2) :=
by
  sorry

end incorrect_statement_D_l38_38751


namespace largest_sampled_item_l38_38718

theorem largest_sampled_item (n : ℕ) (m : ℕ) (a : ℕ) (k : ℕ)
  (hn : n = 360)
  (hm : m = 30)
  (hk : k = n / m)
  (ha : a = 105) :
  ∃ b, b = 433 ∧ (∃ i, i < m ∧ a = 1 + i * k) → (∃ j, j < m ∧ b = 1 + j * k) :=
by
  sorry

end largest_sampled_item_l38_38718


namespace six_digit_palindromes_count_l38_38621

open Nat

theorem six_digit_palindromes_count :
  let digits := {d | 0 ≤ d ∧ d ≤ 9}
  let a_digits := {a | 1 ≤ a ∧ a ≤ 9}
  let b_digits := digits
  let c_digits := digits
  ∃ (total : ℕ), (∀ a ∈ a_digits, ∀ b ∈ b_digits, ∀ c ∈ c_digits, True) → total = 900 :=
by
  sorry

end six_digit_palindromes_count_l38_38621


namespace f_value_2009_l38_38642

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2009
    (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
    (h2 : f 0 ≠ 0) :
    f 2009 = 1 :=
sorry

end f_value_2009_l38_38642


namespace trajectory_equation_l38_38086

def fixed_point : ℝ × ℝ := (1, 2)

def moving_point (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

theorem trajectory_equation (x y : ℝ) (h : dot_product (moving_point x y) fixed_point = 4) :
  x + 2 * y - 4 = 0 :=
sorry

end trajectory_equation_l38_38086


namespace calculate_amount_l38_38809

theorem calculate_amount (p1 p2 p3: ℝ) : 
  p1 = 0.15 * 4000 ∧ 
  p2 = p1 - 0.25 * p1 ∧ 
  p3 = 0.07 * p2 -> 
  (p3 + 0.10 * p3) = 34.65 := 
by 
  sorry

end calculate_amount_l38_38809


namespace divisibility_of_powers_l38_38745

theorem divisibility_of_powers (n : ℤ) : 65 ∣ (7^4 * n - 4^4 * n) :=
by
  sorry

end divisibility_of_powers_l38_38745


namespace inequality_proof_l38_38910

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : (n / m) + (m / n) > 2 :=
sorry

end inequality_proof_l38_38910


namespace trains_meet_in_32_seconds_l38_38931

noncomputable def length_first_train : ℕ := 400
noncomputable def length_second_train : ℕ := 200
noncomputable def initial_distance : ℕ := 200

noncomputable def speed_first_train : ℕ := 15
noncomputable def speed_second_train : ℕ := 10

noncomputable def relative_speed : ℕ := speed_first_train + speed_second_train
noncomputable def total_distance : ℕ := length_first_train + length_second_train + initial_distance
noncomputable def time_to_meet := total_distance / relative_speed

theorem trains_meet_in_32_seconds : time_to_meet = 32 := by
  sorry

end trains_meet_in_32_seconds_l38_38931


namespace balls_into_boxes_l38_38479

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end balls_into_boxes_l38_38479


namespace city_mpg_l38_38365

-- Definitions
def total_distance := 256.2 -- total distance in miles
def total_gallons := 21.0 -- total gallons of gasoline

-- Theorem statement
theorem city_mpg : total_distance / total_gallons = 12.2 :=
by sorry

end city_mpg_l38_38365


namespace find_number_l38_38319

theorem find_number
  (a b c : ℕ)
  (h_a1 : a ≤ 3)
  (h_b1 : b ≤ 3)
  (h_c1 : c ≤ 3)
  (h_a2 : a ≠ 3)
  (h_b_condition1 : b ≠ 1 → 2 * a * b < 10)
  (h_b_condition2 : b ≠ 2 → 2 * a * b < 10)
  (h_c3 : c = 3)
  : a = 2 ∧ b = 3 ∧ c = 3 :=
by
  sorry

end find_number_l38_38319


namespace find_number_l38_38965

theorem find_number (x : ℝ) (h : 0.80 * 40 = (4/5) * x + 16) : x = 20 :=
by sorry

end find_number_l38_38965


namespace distance_from_focus_l38_38608

theorem distance_from_focus (x : ℝ) (A : ℝ × ℝ) (hA_on_parabola : A.1^2 = 4 * A.2) (hA_coord : A.2 = 4) : 
  dist A (0, 1) = 5 := 
by
  sorry

end distance_from_focus_l38_38608


namespace jogging_track_circumference_l38_38710

theorem jogging_track_circumference (speed_deepak speed_wife : ℝ) (time_meet_minutes : ℝ) 
  (h1 : speed_deepak = 20) (h2 : speed_wife = 16) (h3 : time_meet_minutes = 36) : 
  let relative_speed := speed_deepak + speed_wife
  let time_meet_hours := time_meet_minutes / 60
  let circumference := relative_speed * time_meet_hours
  circumference = 21.6 :=
by
  sorry

end jogging_track_circumference_l38_38710


namespace students_with_exactly_two_skills_l38_38670

-- Definitions based on the conditions:
def total_students : ℕ := 150
def students_can_write : ℕ := total_students - 60 -- 150 - 60 = 90
def students_can_direct : ℕ := total_students - 90 -- 150 - 90 = 60
def students_can_produce : ℕ := total_students - 40 -- 150 - 40 = 110

-- The theorem statement
theorem students_with_exactly_two_skills :
  students_can_write + students_can_direct + students_can_produce - total_students = 110 := 
sorry

end students_with_exactly_two_skills_l38_38670


namespace stratified_sampling_example_l38_38733

theorem stratified_sampling_example
  (students_ratio : ℕ → ℕ) -- function to get the number of students in each grade, indexed by natural numbers
  (ratio_cond : students_ratio 0 = 4 ∧ students_ratio 1 = 3 ∧ students_ratio 2 = 2) -- the ratio 4:3:2
  (third_grade_sample : ℕ) -- number of students in the third grade in the sample
  (third_grade_sample_eq : third_grade_sample = 10) -- 10 students from the third grade
  (total_sample_size : ℕ) -- the sample size n
 :
  total_sample_size = 45 := 
sorry

end stratified_sampling_example_l38_38733


namespace probability_distance_greater_than_2_l38_38384

theorem probability_distance_greater_than_2 :
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let area_square := 9
  let area_sector := Real.pi
  let area_shaded := area_square - area_sector
  let P := area_shaded / area_square
  P = (9 - Real.pi) / 9 :=
by
  sorry

end probability_distance_greater_than_2_l38_38384


namespace relation_1_relation_2_relation_3_general_relationship_l38_38932

theorem relation_1 (a b : ℝ) (h1: a = 3) (h2: b = 3) : a^2 + b^2 = 2 * a * b :=
by 
  have h : a = 3 := h1
  have h' : b = 3 := h2
  sorry

theorem relation_2 (a b : ℝ) (h1: a = 2) (h2: b = 1/2) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = 2 := h1
  have h' : b = 1/2 := h2
  sorry

theorem relation_3 (a b : ℝ) (h1: a = -2) (h2: b = 3) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = -2 := h1
  have h' : b = 3 := h2
  sorry

theorem general_relationship (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b :=
by
  sorry

end relation_1_relation_2_relation_3_general_relationship_l38_38932


namespace b_spends_85_percent_l38_38022

-- Definitions based on the given conditions
def combined_salary (a_salary b_salary : ℤ) : Prop := a_salary + b_salary = 3000
def a_salary : ℤ := 2250
def a_spending_ratio : ℝ := 0.95
def a_savings : ℝ := a_salary - a_salary * a_spending_ratio
def b_savings : ℝ := a_savings

-- The goal is to prove that B spends 85% of his salary
theorem b_spends_85_percent (b_salary : ℤ) (b_spending_ratio : ℝ) :
  combined_salary a_salary b_salary →
  b_spending_ratio * b_salary = 0.85 * b_salary :=
  sorry

end b_spends_85_percent_l38_38022


namespace square_area_increase_l38_38549

variable (a : ℕ)

theorem square_area_increase (a : ℕ) :
  (a + 6) ^ 2 - a ^ 2 = 12 * a + 36 :=
by
  sorry

end square_area_increase_l38_38549


namespace rectangle_length_l38_38448

theorem rectangle_length (P W : ℝ) (hP : P = 30) (hW : W = 10) :
  ∃ (L : ℝ), 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end rectangle_length_l38_38448


namespace num_neg_values_of_x_l38_38343

theorem num_neg_values_of_x 
  (n : ℕ) 
  (xn_pos_int : ∃ k, n = k ∧ k > 0) 
  (sqrt_x_169_pos_int : ∀ x, ∃ m, x + 169 = m^2 ∧ m > 0) :
  ∃ count, count = 12 := 
by
  sorry

end num_neg_values_of_x_l38_38343


namespace find_x_l38_38682

-- Definitions for the angles
def angle1 (x : ℝ) := 3 * x
def angle2 (x : ℝ) := 7 * x
def angle3 (x : ℝ) := 4 * x
def angle4 (x : ℝ) := 2 * x
def angle5 (x : ℝ) := x

-- The condition that the sum of the angles equals 360 degrees
def sum_of_angles (x : ℝ) := angle1 x + angle2 x + angle3 x + angle4 x + angle5 x = 360

-- The statement to prove
theorem find_x (x : ℝ) (hx : sum_of_angles x) : x = 360 / 17 := by
  -- Proof to be written here
  sorry

end find_x_l38_38682


namespace amusement_park_total_cost_l38_38699

def rides_cost_ferris_wheel : ℕ := 5 * 6
def rides_cost_roller_coaster : ℕ := 7 * 4
def rides_cost_merry_go_round : ℕ := 3 * 10
def rides_cost_bumper_cars : ℕ := 4 * 7
def rides_cost_haunted_house : ℕ := 6 * 5
def rides_cost_log_flume : ℕ := 8 * 3

def snacks_cost_ice_cream : ℕ := 8 * 4
def snacks_cost_hot_dog : ℕ := 6 * 5
def snacks_cost_pizza : ℕ := 4 * 3
def snacks_cost_pretzel : ℕ := 5 * 2
def snacks_cost_cotton_candy : ℕ := 3 * 6
def snacks_cost_soda : ℕ := 2 * 7

def total_rides_cost : ℕ := 
  rides_cost_ferris_wheel + 
  rides_cost_roller_coaster + 
  rides_cost_merry_go_round + 
  rides_cost_bumper_cars + 
  rides_cost_haunted_house + 
  rides_cost_log_flume

def total_snacks_cost : ℕ := 
  snacks_cost_ice_cream + 
  snacks_cost_hot_dog + 
  snacks_cost_pizza + 
  snacks_cost_pretzel + 
  snacks_cost_cotton_candy + 
  snacks_cost_soda

def total_cost : ℕ :=
  total_rides_cost + total_snacks_cost

theorem amusement_park_total_cost :
  total_cost = 286 :=
by
  unfold total_cost total_rides_cost total_snacks_cost
  unfold rides_cost_ferris_wheel 
         rides_cost_roller_coaster 
         rides_cost_merry_go_round 
         rides_cost_bumper_cars 
         rides_cost_haunted_house 
         rides_cost_log_flume
         snacks_cost_ice_cream 
         snacks_cost_hot_dog 
         snacks_cost_pizza 
         snacks_cost_pretzel 
         snacks_cost_cotton_candy 
         snacks_cost_soda
  sorry

end amusement_park_total_cost_l38_38699


namespace min_value_reciprocal_sum_l38_38537

theorem min_value_reciprocal_sum (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_sum : x + y = 1) : 
  ∃ z, z = 4 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 -> z ≤ (1/x + 1/y)) :=
sorry

end min_value_reciprocal_sum_l38_38537


namespace positive_rational_representation_l38_38973

theorem positive_rational_representation (q : ℚ) (h_pos_q : 0 < q) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by
  sorry

end positive_rational_representation_l38_38973


namespace bob_password_probability_l38_38555

def num_non_negative_single_digits : ℕ := 10
def num_odd_single_digits : ℕ := 5
def num_even_positive_single_digits : ℕ := 4
def probability_first_digit_odd : ℚ := num_odd_single_digits / num_non_negative_single_digits
def probability_middle_letter : ℚ := 1
def probability_last_digit_even_positive : ℚ := num_even_positive_single_digits / num_non_negative_single_digits

theorem bob_password_probability :
  probability_first_digit_odd * probability_middle_letter * probability_last_digit_even_positive = 1 / 5 :=
by
  sorry

end bob_password_probability_l38_38555


namespace total_people_going_to_zoo_l38_38387

def cars : ℝ := 3.0
def people_per_car : ℝ := 63.0

theorem total_people_going_to_zoo : cars * people_per_car = 189.0 :=
by 
  sorry

end total_people_going_to_zoo_l38_38387


namespace points_on_line_l38_38746

theorem points_on_line (y1 y2 : ℝ) 
  (hA : y1 = - (1 / 2 : ℝ) * 1 - 1) 
  (hB : y2 = - (1 / 2 : ℝ) * 3 - 1) :
  y1 > y2 := 
by
  sorry

end points_on_line_l38_38746


namespace solve_variable_expression_l38_38819

variable {x y : ℕ}

theorem solve_variable_expression
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (7 * x + 5 * y) / (x - 2 * y) = 26) :
  x = 3 * y :=
sorry

end solve_variable_expression_l38_38819


namespace geometric_mean_a_b_l38_38976

theorem geometric_mean_a_b : ∀ (a b : ℝ), a > 0 → b > 0 → Real.sqrt 3 = Real.sqrt (3^a * 3^b) → a + b = 1 :=
by
  intros a b ha hb hgeo
  sorry

end geometric_mean_a_b_l38_38976


namespace a5_eq_11_l38_38340

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ) (d : ℚ) (a1 : ℚ)

-- The definitions as given in the conditions
def arithmetic_sequence (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_of_terms (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def cond1 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 3 + S 3 = 22

def cond2 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 4 - S 4 = -15

-- The statement to prove
theorem a5_eq_11 (a : ℕ → ℚ) (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a a1 d)
  (h_sum : sum_of_terms S a1 d)
  (h1 : cond1 a S)
  (h2 : cond2 a S) : a 5 = 11 := by
  sorry

end a5_eq_11_l38_38340


namespace min_value_of_parabola_l38_38177

theorem min_value_of_parabola : ∃ x : ℝ, ∀ y : ℝ, y = 3 * x^2 - 18 * x + 244 → y = 217 := by
  sorry

end min_value_of_parabola_l38_38177


namespace simplify_fraction_l38_38513

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l38_38513


namespace radius_first_field_l38_38461

theorem radius_first_field (r_2 : ℝ) (h_r2 : r_2 = 10) (h_area : ∃ A_2, ∃ A_1, A_1 = 0.09 * A_2 ∧ A_2 = π * r_2^2) : ∃ r_1 : ℝ, r_1 = 3 :=
by
  sorry

end radius_first_field_l38_38461


namespace waiting_time_probability_l38_38011

theorem waiting_time_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t < 30 → (1 / 30) * (if t < 25 then 5 else 5 - (t - 25)) = 1 / 6) :=
by
  sorry

end waiting_time_probability_l38_38011


namespace original_number_is_17_l38_38865

-- Function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  (ones * 10) + tens

-- Problem statement
theorem original_number_is_17 (x : ℕ) (h1 : reverse_digits (2 * x) + 2 = 45) : x = 17 :=
by
  sorry

end original_number_is_17_l38_38865


namespace max_value_of_a_l38_38034

variable {a : ℝ}

theorem max_value_of_a (h : a > 0) : 
  (∀ x : ℝ, x > 0 → (2 * x^2 - a * x + a > 0)) ↔ a ≤ 8 := 
sorry

end max_value_of_a_l38_38034


namespace constant_term_g_eq_l38_38234

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

theorem constant_term_g_eq : 
  (h.coeff 0 = 2) ∧ (f.coeff 0 = -6) →  g.coeff 0 = -1/3 := by
  sorry

end constant_term_g_eq_l38_38234


namespace final_remaining_money_l38_38833

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end final_remaining_money_l38_38833


namespace max_glows_in_time_range_l38_38160

theorem max_glows_in_time_range (start_time end_time : ℤ) (interval : ℤ) (h1 : start_time = 3600 + 3420 + 58) (h2 : end_time = 10800 + 1200 + 47) (h3 : interval = 21) :
  (end_time - start_time) / interval = 236 := 
  sorry

end max_glows_in_time_range_l38_38160


namespace tangent_lines_through_origin_l38_38780

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

variable (a : ℝ)

theorem tangent_lines_through_origin 
  (h1 : ∃ m1 m2 : ℝ, m1 ≠ m2 ∧ (f a (-m1) + f a (m1 + 2)) / 2 = f a 1) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (f a t1 * (1 / t1) = f a 0) ∧ (f a t2 * (1 / t2) = f a 0) := 
sorry

end tangent_lines_through_origin_l38_38780


namespace max_dance_counts_possible_l38_38236

noncomputable def max_dance_counts : ℕ := 29

theorem max_dance_counts_possible (boys girls : ℕ) (dance_count : ℕ → ℕ) :
   boys = 29 → girls = 15 → 
   (∀ b, b < boys → dance_count b ≤ girls) → 
   (∀ g, g < girls → ∃ d, d ≤ boys ∧ dance_count d = g) →
   (∃ d, d ≤ max_dance_counts ∧
     (∀ k, k ≤ d → (∃ b, b < boys ∧ dance_count b = k) ∨ (∃ g, g < girls ∧ dance_count g = k))) := 
sorry

end max_dance_counts_possible_l38_38236


namespace math_problem_l38_38827

theorem math_problem
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2006)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2007)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2006)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2007)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2006)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2007)
  : (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = -1 / 2006 := by
  sorry

end math_problem_l38_38827


namespace flying_scotsman_more_carriages_l38_38258

theorem flying_scotsman_more_carriages :
  ∀ (E N No F T D : ℕ),
    E = 130 →
    E = N + 20 →
    No = 100 →
    T = 460 →
    D = F - No →
    F + E + N + No = T →
    D = 20 :=
by
  intros E N No F T D hE1 hE2 hNo hT hD hSum
  sorry

end flying_scotsman_more_carriages_l38_38258


namespace math_problem_l38_38569

theorem math_problem
  (x y : ℚ)
  (h1 : x + y = 11 / 17)
  (h2 : x - y = 1 / 143) :
  x^2 - y^2 = 11 / 2431 :=
by
  sorry

end math_problem_l38_38569


namespace compare_powers_l38_38713

-- Definitions for the three numbers
def a : ℝ := 3 ^ 555
def b : ℝ := 4 ^ 444
def c : ℝ := 5 ^ 333

-- Statement to prove
theorem compare_powers : c < a ∧ a < b := sorry

end compare_powers_l38_38713


namespace vector_dot_product_example_l38_38306

noncomputable def vector_dot_product (e1 e2 : ℝ) : ℝ :=
  let c := e1 * (-3 * e1)
  let d := (e1 * (2 * e2))
  let e := (e2 * (2 * e2))
  c + d + e

theorem vector_dot_product_example (e1 e2 : ℝ) (unit_vectors : e1^2 = 1 ∧ e2^2 = 1) :
  (e1 - e2) * (e1 - e2) = 1 ∧ (e1 * e2 = 1 / 2) → 
  vector_dot_product e1 e2 = -5 / 2 := by {
  sorry
}

end vector_dot_product_example_l38_38306


namespace quotient_is_12_l38_38698

theorem quotient_is_12 (a b q : ℕ) (h1: q = a / b) (h2: q = a / 2) (h3: q = 6 * b) : q = 12 :=
by 
  sorry

end quotient_is_12_l38_38698


namespace addition_addends_l38_38191

theorem addition_addends (a b : ℕ) (c₁ c₂ : ℕ) (d : ℕ) : 
  a + b = c₁ ∧ a + (b - d) = c₂ ∧ d = 50 ∧ c₁ = 982 ∧ c₂ = 577 → 
  a = 450 ∧ b = 532 :=
by
  sorry

end addition_addends_l38_38191


namespace dice_digit_distribution_l38_38740

theorem dice_digit_distribution : ∃ n : ℕ, n = 10 ∧ 
  (∀ (d1 d2 : Finset ℕ), d1.card = 6 ∧ d2.card = 6 ∧
  (0 ∈ d1) ∧ (1 ∈ d1) ∧ (2 ∈ d1) ∧ 
  (0 ∈ d2) ∧ (1 ∈ d2) ∧ (2 ∈ d2) ∧
  ({3, 4, 5, 6, 7, 8} ⊆ (d1 ∪ d2)) ∧ 
  (∀ i, i ∈ d1 ∪ d2 → i ∈ (Finset.range 10))) := 
  sorry

end dice_digit_distribution_l38_38740


namespace find_b_l38_38638

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def derivative_at_one (a : ℝ) : ℝ := a + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem find_b (a b : ℝ) (h_deriv : derivative_at_one a = 2) (h_tangent : tangent_line b 1 = curve a 1) :
  b = -1 :=
by
  sorry

end find_b_l38_38638


namespace smallest_four_digit_multiple_of_8_with_digit_sum_20_l38_38140

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.foldl (· + ·) 0

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20:
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ sum_of_digits n = 20 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 8 = 0 ∧ sum_of_digits m = 20 → n ≤ m :=
by { sorry }

end smallest_four_digit_multiple_of_8_with_digit_sum_20_l38_38140


namespace shirts_sold_l38_38406

theorem shirts_sold (initial final : ℕ) (h : initial = 49) (h1 : final = 28) : initial - final = 21 :=
sorry

end shirts_sold_l38_38406


namespace term_5th_in_sequence_l38_38924

theorem term_5th_in_sequence : 
  ∃ n : ℕ, n = 5 ∧ ( ∃ t : ℕ, t = 28 ∧ 3^t ∈ { 3^(7 * (k - 1)) | k : ℕ } ) :=
by {
  sorry
}

end term_5th_in_sequence_l38_38924


namespace part_I_part_II_l38_38276

noncomputable def seq_a : ℕ → ℝ 
| 0       => 1   -- Normally, we start with n = 1, so we set a_0 to some default value.
| (n+1)   => (1 + 1 / (n^2 + n)) * seq_a n + 1 / (2^n)

theorem part_I (n : ℕ) (h: n ≥ 2) : seq_a n ≥ 2 :=
sorry

theorem part_II (n : ℕ) : seq_a n < Real.exp 2 :=
sorry

-- Assumption: ln(1 + x) < x for all x > 0
axiom ln_ineq (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x

end part_I_part_II_l38_38276


namespace rectangle_pentagon_ratio_l38_38960

theorem rectangle_pentagon_ratio
  (l w p : ℝ)
  (h1 : l = 2 * w)
  (h2 : 2 * (l + w) = 30)
  (h3 : 5 * p = 30) :
  l / p = 5 / 3 :=
by
  sorry

end rectangle_pentagon_ratio_l38_38960


namespace sufficient_not_necessary_condition_l38_38165

theorem sufficient_not_necessary_condition (a : ℝ)
  : (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, a ≥ 0 ∨ a * x^2 + x + 1 ≥ 0)
:= sorry

end sufficient_not_necessary_condition_l38_38165


namespace minimum_a2_plus_4b2_l38_38721

theorem minimum_a2_plus_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) : 
  a^2 + 4 * b^2 ≥ 32 :=
sorry

end minimum_a2_plus_4b2_l38_38721


namespace mart_income_percentage_j_l38_38038

variables (J T M : ℝ)

-- condition: Tim's income is 40 percent less than Juan's income
def tims_income := T = 0.60 * J

-- condition: Mart's income is 40 percent more than Tim's income
def marts_income := M = 1.40 * T

-- goal: Prove that Mart's income is 84 percent of Juan's income
theorem mart_income_percentage_j (J : ℝ) (T : ℝ) (M : ℝ)
  (h1 : T = 0.60 * J) 
  (h2 : M = 1.40 * T) : 
  M = 0.84 * J := 
sorry

end mart_income_percentage_j_l38_38038


namespace celina_paid_multiple_of_diego_l38_38263

theorem celina_paid_multiple_of_diego
  (D : ℕ) (x : ℕ)
  (h_total : (x + 1) * D + 1000 = 50000)
  (h_positive : D > 0) :
  x = 48 :=
sorry

end celina_paid_multiple_of_diego_l38_38263


namespace probability_at_least_one_head_and_die_3_l38_38242

-- Define the probability of an event happening
noncomputable def probability_of_event (total_outcomes : ℕ) (successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

-- Define the problem specific values
def total_coin_outcomes : ℕ := 4
def successful_coin_outcomes : ℕ := 3
def total_die_outcomes : ℕ := 8
def successful_die_outcome : ℕ := 1
def total_outcomes : ℕ := total_coin_outcomes * total_die_outcomes
def successful_outcomes : ℕ := successful_coin_outcomes * successful_die_outcome

-- Prove that the probability of at least one head in two coin flips and die showing a 3 is 3/32
theorem probability_at_least_one_head_and_die_3 : 
  probability_of_event total_outcomes successful_outcomes = 3 / 32 := by
  sorry

end probability_at_least_one_head_and_die_3_l38_38242


namespace maximum_of_function_l38_38979

theorem maximum_of_function :
  ∃ x y : ℝ, 
    (1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/4 ≤ y ∧ y ≤ 5/12) ∧ 
    (∀ x' y' : ℝ, 1/3 ≤ x' ∧ x' ≤ 2/5 ∧ 1/4 ≤ y' ∧ y' ≤ 5/12 → 
                (xy / (x^2 + y^2) ≤ x' * y' / (x'^2 + y'^2))) ∧ 
    (xy / (x^2 + y^2) = 20 / 41) := 
sorry

end maximum_of_function_l38_38979


namespace how_many_integers_satisfy_l38_38408

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l38_38408


namespace distribute_paper_clips_l38_38032

theorem distribute_paper_clips (total_paper_clips boxes : ℕ) (h_total : total_paper_clips = 81) (h_boxes : boxes = 9) : total_paper_clips / boxes = 9 := by
  sorry

end distribute_paper_clips_l38_38032


namespace longer_piece_length_l38_38831

-- Conditions
def total_length : ℤ := 69
def is_cuts_into_two_pieces (a b : ℤ) : Prop := a + b = total_length
def is_twice_the_length (a b : ℤ) : Prop := a = 2 * b

-- Question: What is the length of the longer piece?
theorem longer_piece_length
  (a b : ℤ) 
  (H1: is_cuts_into_two_pieces a b)
  (H2: is_twice_the_length a b) :
  a = 46 :=
sorry

end longer_piece_length_l38_38831


namespace compute_difference_of_squares_l38_38841

theorem compute_difference_of_squares :
  let a := 23
  let b := 12
  (a + b) ^ 2 - (a - b) ^ 2 = 1104 := by
sorry

end compute_difference_of_squares_l38_38841


namespace miles_total_instruments_l38_38798

-- Definitions based on the conditions
def fingers : ℕ := 10
def hands : ℕ := 2
def heads : ℕ := 1
def trumpets : ℕ := fingers - 3
def guitars : ℕ := hands + 2
def trombones : ℕ := heads + 2
def french_horns : ℕ := guitars - 1
def total_instruments : ℕ := trumpets + guitars + trombones + french_horns

-- Main theorem
theorem miles_total_instruments : total_instruments = 17 := 
sorry

end miles_total_instruments_l38_38798


namespace odd_function_f_l38_38969

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_f (f_odd : ∀ x : ℝ, f (-x) = - f x)
                       (f_lt_0 : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = - x * (x + 1) :=
by
  sorry

end odd_function_f_l38_38969


namespace lally_internet_days_l38_38064

-- Definitions based on the conditions
def cost_per_day : ℝ := 0.5
def debt_limit : ℝ := 5
def initial_payment : ℝ := 7
def initial_balance : ℝ := 0

-- Proof problem statement
theorem lally_internet_days : ∀ (d : ℕ), 
  (initial_balance + initial_payment - cost_per_day * d ≤ debt_limit) -> (d = 14) :=
sorry

end lally_internet_days_l38_38064


namespace exist_m_n_l38_38018

theorem exist_m_n (p : ℕ) [hp : Fact (Nat.Prime p)] (h : 5 < p) :
  ∃ m n : ℕ, (m + n < p ∧ p ∣ (2^m * 3^n - 1)) := sorry

end exist_m_n_l38_38018


namespace find_theta_l38_38254

theorem find_theta (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x ^ 3 * Real.sin θ + x ^ 2 * Real.cos θ - x * (1 - x) + (1 - x) ^ 2 * Real.sin θ > 0) → 
  Real.sin θ > 0 → 
  Real.cos θ + Real.sin θ > 0 → 
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intro θ_range all_x_condition sin_pos cos_sin_pos
  sorry

end find_theta_l38_38254


namespace find_ratio_l38_38876

variable {d : ℕ}
variable {a : ℕ → ℝ}

-- Conditions: arithmetic sequence with non-zero common difference, and geometric sequence terms
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d
axiom non_zero_d : d ≠ 0
axiom geometric_sequence : (a 1 + 2*d)^2 = a 1 * (a 1 + 8*d)

-- Theorem to prove the desired ratio
theorem find_ratio : (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
sorry

end find_ratio_l38_38876


namespace largest_gcd_of_sum_1729_l38_38047

theorem largest_gcd_of_sum_1729 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1729) :
  ∃ g, g = Nat.gcd x y ∧ g = 247 := sorry

end largest_gcd_of_sum_1729_l38_38047


namespace correct_reference_l38_38527

variable (house : String) 
variable (beautiful_garden_in_front : Bool)
variable (I_like_this_house : Bool)
variable (enough_money_to_buy : Bool)

-- Statement: Given the conditions, prove that the correct word to fill in the blank is "it".
theorem correct_reference : I_like_this_house ∧ beautiful_garden_in_front ∧ ¬ enough_money_to_buy → "it" = "correct choice" :=
by
  sorry

end correct_reference_l38_38527


namespace tangent_line_at_point_l38_38320

noncomputable def curve (x : ℝ) : ℝ := Real.exp x + x

theorem tangent_line_at_point :
  (∃ k b : ℝ, (∀ x : ℝ, curve x = k * x + b) ∧ k = 2 ∧ b = 1) :=
by
  sorry

end tangent_line_at_point_l38_38320


namespace fraction_q_over_p_l38_38530

noncomputable def proof_problem (p q : ℝ) : Prop :=
  ∃ k : ℝ, p = 9^k ∧ q = 12^k ∧ p + q = 16^k

theorem fraction_q_over_p (p q : ℝ) (h : proof_problem p q) : q / p = (1 + Real.sqrt 5) / 2 :=
sorry

end fraction_q_over_p_l38_38530


namespace domain_of_expression_l38_38404

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l38_38404


namespace sq_diff_eq_binom_identity_l38_38376

variable (a b : ℝ)

theorem sq_diff_eq_binom_identity : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 :=
by
  sorry

end sq_diff_eq_binom_identity_l38_38376


namespace remainder_when_13_add_x_div_31_eq_22_l38_38186

open BigOperators

theorem remainder_when_13_add_x_div_31_eq_22
  (x : ℕ) (hx : x > 0) (hmod : 7 * x ≡ 1 [MOD 31]) :
  (13 + x) % 31 = 22 := 
  sorry

end remainder_when_13_add_x_div_31_eq_22_l38_38186


namespace exists_N_with_N_and_N2_ending_same_l38_38380

theorem exists_N_with_N_and_N2_ending_same : 
  ∃ (N : ℕ), (N > 0) ∧ (N % 100000 = (N*N) % 100000) ∧ (N / 10000 ≠ 0) := sorry

end exists_N_with_N_and_N2_ending_same_l38_38380


namespace required_speed_l38_38994

theorem required_speed
  (D T : ℝ) (h1 : 30 = D / T) 
  (h2 : 2 * D / 3 = 30 * (T / 3)) :
  (D / 3) / (2 * T / 3) = 15 :=
by
  sorry

end required_speed_l38_38994


namespace rectangle_diagonals_not_perpendicular_l38_38104

-- Definition of a rectangle through its properties
structure Rectangle (α : Type _) [LinearOrderedField α] :=
  (angle_eq : ∀ (a : α), a = 90)
  (diagonals_eq : ∀ (d1 d2 : α), d1 = d2)
  (diagonals_bisect : ∀ (d1 d2 : α), d1 / 2 = d2 / 2)

-- Theorem stating that a rectangle's diagonals are not necessarily perpendicular
theorem rectangle_diagonals_not_perpendicular (α : Type _) [LinearOrderedField α] (R : Rectangle α) : 
  ¬ (∀ (d1 d2 : α), d1 * d2 = 0) :=
sorry

end rectangle_diagonals_not_perpendicular_l38_38104


namespace shanghai_masters_total_matches_l38_38756

theorem shanghai_masters_total_matches : 
  let players := 8
  let groups := 2
  let players_per_group := 4
  let round_robin_matches_per_group := (players_per_group * (players_per_group - 1)) / 2
  let round_robin_total_matches := round_robin_matches_per_group * groups
  let elimination_matches := 2 * (groups - 1)  -- semi-final matches
  let final_matches := 2  -- one final and one third-place match
  round_robin_total_matches + elimination_matches + final_matches = 16 :=
by
  sorry

end shanghai_masters_total_matches_l38_38756


namespace sarahs_total_problems_l38_38462

def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def science_pages : ℕ := 5
def math_problems_per_page : ℕ := 4
def reading_problems_per_page : ℕ := 4
def science_problems_per_page : ℕ := 6

def total_math_problems : ℕ := math_pages * math_problems_per_page
def total_reading_problems : ℕ := reading_pages * reading_problems_per_page
def total_science_problems : ℕ := science_pages * science_problems_per_page

def total_problems : ℕ := total_math_problems + total_reading_problems + total_science_problems

theorem sarahs_total_problems :
  total_problems = 70 :=
by
  -- proof will be inserted here
  sorry

end sarahs_total_problems_l38_38462


namespace not_exists_odd_product_sum_l38_38157

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l38_38157


namespace total_reams_l38_38119

theorem total_reams (h_r : ℕ) (s_r : ℕ) : h_r = 2 → s_r = 3 → h_r + s_r = 5 :=
by
  intro h_eq s_eq
  sorry

end total_reams_l38_38119


namespace sum_of_three_integers_eq_57_l38_38324

theorem sum_of_three_integers_eq_57
  (a b c : ℕ) (h1: a * b * c = 7^3) (h2: a ≠ b) (h3: b ≠ c) (h4: a ≠ c) :
  a + b + c = 57 :=
sorry

end sum_of_three_integers_eq_57_l38_38324


namespace express_y_in_terms_of_x_l38_38183

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 5) : y = 2 * x + 5 :=
by
  sorry

end express_y_in_terms_of_x_l38_38183


namespace angle_measure_l38_38229

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l38_38229


namespace af_over_cd_is_025_l38_38862

theorem af_over_cd_is_025
  (a b c d e f X : ℝ)
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end af_over_cd_is_025_l38_38862


namespace expenditure_recorded_neg_20_l38_38220

-- Define the condition where income of 60 yuan is recorded as +60 yuan
def income_recorded (income : ℤ) : ℤ :=
  income

-- Define what expenditure is given the condition
def expenditure_recorded (expenditure : ℤ) : ℤ :=
  -expenditure

-- Prove that an expenditure of 20 yuan is recorded as -20 yuan
theorem expenditure_recorded_neg_20 :
  expenditure_recorded 20 = -20 :=
by
  sorry

end expenditure_recorded_neg_20_l38_38220


namespace largest_four_digit_number_divisible_by_4_with_digit_sum_20_l38_38748

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def digit_sum_is_20 (n : ℕ) : Prop :=
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 20

theorem largest_four_digit_number_divisible_by_4_with_digit_sum_20 :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_4 n ∧ digit_sum_is_20 n ∧ ∀ m : ℕ, is_four_digit m ∧ is_divisible_by_4 m ∧ digit_sum_is_20 m → m ≤ n :=
  sorry

end largest_four_digit_number_divisible_by_4_with_digit_sum_20_l38_38748


namespace no_such_function_exists_l38_38192

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = f (n + 1) - f n :=
by
  sorry

end no_such_function_exists_l38_38192


namespace find_f_neg_8point5_l38_38367

def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom initial_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_neg_8point5 : f (-8.5) = -0.5 :=
by
  -- Expect this proof to follow the outlined logic
  sorry

end find_f_neg_8point5_l38_38367


namespace lara_sees_leo_for_six_minutes_l38_38657

-- Define constants for speeds and initial distances
def lara_speed : ℕ := 60
def leo_speed : ℕ := 40
def initial_distance : ℕ := 1
def time_to_minutes (t : ℚ) : ℚ := t * 60
-- Define the condition that proves Lara can see Leo for 6 minutes
theorem lara_sees_leo_for_six_minutes :
  lara_speed > leo_speed ∧
  initial_distance > 0 ∧
  (initial_distance : ℚ) / (lara_speed - leo_speed) * 2 = (6 : ℚ) / 60 :=
by
  sorry

end lara_sees_leo_for_six_minutes_l38_38657


namespace tangent_line_condition_l38_38720

theorem tangent_line_condition (k : ℝ) : 
  (∀ x y : ℝ, (x-2)^2 + (y-1)^2 = 1 → x - k * y - 1 = 0 → False) ↔ k = 0 :=
sorry

end tangent_line_condition_l38_38720


namespace area_of_region_enclosed_by_graph_l38_38351

noncomputable def area_of_enclosed_region : ℝ :=
  let x1 := 41.67
  let x2 := 62.5
  let y1 := 8.33
  let y2 := -8.33
  0.5 * (x2 - x1) * (y1 - y2)

theorem area_of_region_enclosed_by_graph :
  area_of_enclosed_region = 173.28 :=
sorry

end area_of_region_enclosed_by_graph_l38_38351


namespace smallest_possible_x_l38_38544

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l38_38544


namespace find_m_n_find_a_l38_38009

def quadratic_roots (x : ℝ) (m n : ℝ) : Prop := 
  x^2 + m * x - 3 = 0

theorem find_m_n {m n : ℝ} : 
  quadratic_roots (-1) m n ∧ quadratic_roots n m n → 
  m = -2 ∧ n = 3 := 
sorry

def f (x m : ℝ) : ℝ := 
  x^2 + m * x - 3

theorem find_a {a m : ℝ} (h : m = -2) : 
  f 3 m = f (2 * a - 3) m → 
  a = 1 ∨ a = 3 := 
sorry

end find_m_n_find_a_l38_38009


namespace vector_at_t_neg3_l38_38787

theorem vector_at_t_neg3 :
  let a := (2, 3)
  let b := (12, -37)
  let d := ((b.1 - a.1) / 5, (b.2 - a.2) / 5)
  let line_param (t : ℝ) := (a.1 + t * d.1, a.2 + t * d.2)
  line_param (-3) = (-4, 27) := by
  -- Proof goes here
  sorry

end vector_at_t_neg3_l38_38787


namespace problem_l38_38674

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

def p : Prop := ∀ x : ℝ, x ≠ 0 → f x ≥ 4 ∧ (∃ x : ℝ, x > 0 ∧ f x = 4)

def q : Prop := ∀ (A B C : ℝ) (a b c : ℝ),
  A > B ↔ a > b

theorem problem : (¬p) ∧ q :=
sorry

end problem_l38_38674


namespace sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l38_38299

noncomputable def b : ℕ → ℚ
| 0     => 2
| 1     => 3
| (n+2) => 2 * b (n+1) + 3 * b n

theorem sum_bn_over_3_pow_n_plus_1_eq_2_over_5 :
  (∑' n : ℕ, (b n) / (3 ^ (n + 1))) = (2 / 5) :=
by
  sorry

end sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l38_38299


namespace determine_b_l38_38811

theorem determine_b (a b : ℤ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 :=
by
  sorry

end determine_b_l38_38811


namespace general_formula_for_sequence_l38_38506

theorem general_formula_for_sequence :
  ∀ (a : ℕ → ℕ), (a 0 = 1) → (a 1 = 1) →
  (∀ n, 2 ≤ n → a n = 2 * a (n - 1) - a (n - 2)) →
  ∀ n, a n = (2^n - 1)^2 :=
by
  sorry

end general_formula_for_sequence_l38_38506


namespace sequence_first_five_terms_l38_38941

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n + (n : ℤ)

theorem sequence_first_five_terms :
  a_n 1 = 0 ∧
  a_n 2 = 3 ∧
  a_n 3 = 2 ∧
  a_n 4 = 5 ∧
  a_n 5 = 4 :=
by
  sorry

end sequence_first_five_terms_l38_38941


namespace min_eq_one_implies_x_eq_one_l38_38677

open Real

theorem min_eq_one_implies_x_eq_one (x : ℝ) (h : min (1/2 + x) (x^2) = 1) : x = 1 := 
sorry

end min_eq_one_implies_x_eq_one_l38_38677


namespace find_natural_numbers_l38_38381

-- Problem statement: Find all natural numbers x, y, z such that 3^x + 4^y = 5^z
theorem find_natural_numbers (x y z : ℕ) (h : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

end find_natural_numbers_l38_38381


namespace karen_box_crayons_l38_38650

theorem karen_box_crayons (judah_crayons : ℕ) (gilbert_crayons : ℕ) (beatrice_crayons : ℕ) (karen_crayons : ℕ)
  (h1 : judah_crayons = 8)
  (h2 : gilbert_crayons = 4 * judah_crayons)
  (h3 : beatrice_crayons = 2 * gilbert_crayons)
  (h4 : karen_crayons = 2 * beatrice_crayons) :
  karen_crayons = 128 :=
by
  sorry

end karen_box_crayons_l38_38650


namespace rent_increase_l38_38253

theorem rent_increase (monthly_rent_first_3_years : ℕ) (months_first_3_years : ℕ) 
  (total_paid : ℕ) (total_years : ℕ) (months_in_a_year : ℕ) (new_monthly_rent : ℕ) :
  monthly_rent_first_3_years * (months_in_a_year * 3) + new_monthly_rent * (months_in_a_year * (total_years - 3)) = total_paid →
  new_monthly_rent = 350 :=
by
  intros h
  -- proof development
  sorry

end rent_increase_l38_38253


namespace ice_cream_eaten_on_friday_l38_38856

theorem ice_cream_eaten_on_friday
  (x : ℝ) -- the amount eaten on Friday night
  (saturday_night : ℝ) -- the amount eaten on Saturday night
  (total : ℝ) -- the total amount eaten
  
  (h1 : saturday_night = 0.25)
  (h2 : total = 3.5)
  (h3 : x + saturday_night = total) : x = 3.25 :=
by
  sorry

end ice_cream_eaten_on_friday_l38_38856


namespace possible_distances_AG_l38_38358

theorem possible_distances_AG (A B V G : ℝ) (AB VG : ℝ) (x AG : ℝ) :
  (AB = 600) →
  (VG = 600) →
  (AG = 3 * x) →
  (AG = 900 ∨ AG = 1800) :=
by
  intros h1 h2 h3
  sorry

end possible_distances_AG_l38_38358


namespace quadratic_y_axis_intersection_l38_38958

theorem quadratic_y_axis_intersection :
  (∃ y, (y = (0 - 1) ^ 2 + 2) ∧ (0, y) = (0, 3)) :=
sorry

end quadratic_y_axis_intersection_l38_38958


namespace Vitya_catches_mother_l38_38442

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l38_38442


namespace range_of_m_l38_38252

noncomputable def quadratic_polynomial (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + m^2 - 2

theorem range_of_m (m : ℝ) (h1 : ∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ quadratic_polynomial m x1 = 0 ∧ quadratic_polynomial m x2 = 0) :
  0 < m ∧ m < 1 :=
sorry

end range_of_m_l38_38252


namespace extreme_point_at_1_l38_38317

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (2 * a^3 - a^2) * Real.log x - (a^2 + 2 * a - 1) * x

theorem extreme_point_at_1 (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ ∀ x > 0, deriv (f a) x = 0 →
  a = -1) := sorry

end extreme_point_at_1_l38_38317


namespace smallest_positive_integer_cube_ends_544_l38_38185

theorem smallest_positive_integer_cube_ends_544 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 544 → m ≥ n :=
by
  sorry

end smallest_positive_integer_cube_ends_544_l38_38185


namespace julia_paint_area_l38_38176

noncomputable def area_to_paint (bedroom_length: ℕ) (bedroom_width: ℕ) (bedroom_height: ℕ) (non_paint_area: ℕ) (num_bedrooms: ℕ) : ℕ :=
  let wall_area_one_bedroom := 2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)
  let paintable_area_one_bedroom := wall_area_one_bedroom - non_paint_area
  num_bedrooms * paintable_area_one_bedroom

theorem julia_paint_area :
  area_to_paint 14 11 9 70 4 = 1520 :=
by
  sorry

end julia_paint_area_l38_38176


namespace fraction_increase_l38_38562

theorem fraction_increase (m n a : ℕ) (h1 : m > n) (h2 : a > 0) : 
  (n : ℚ) / m < (n + a : ℚ) / (m + a) :=
by
  sorry

end fraction_increase_l38_38562


namespace abs_ab_eq_2_sqrt_111_l38_38049

theorem abs_ab_eq_2_sqrt_111 (a b : ℝ) (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) : |a * b| = 2 * Real.sqrt 111 := sorry

end abs_ab_eq_2_sqrt_111_l38_38049


namespace sum_xyz_is_sqrt_13_l38_38860

variable (x y z : ℝ)

-- The conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z

axiom eq1 : x^2 + y^2 + x * y = 3
axiom eq2 : y^2 + z^2 + y * z = 4
axiom eq3 : z^2 + x^2 + z * x = 7 

-- The theorem statement: Prove that x + y + z = sqrt(13)
theorem sum_xyz_is_sqrt_13 : x + y + z = Real.sqrt 13 :=
by
  sorry

end sum_xyz_is_sqrt_13_l38_38860


namespace correct_mean_l38_38285

-- Definitions of conditions
def n : ℕ := 30
def mean_incorrect : ℚ := 140
def value_correct : ℕ := 145
def value_incorrect : ℕ := 135

-- The statement to be proved
theorem correct_mean : 
  let S_incorrect := mean_incorrect * n
  let Difference := value_correct - value_incorrect
  let S_correct := S_incorrect + Difference
  let mean_correct := S_correct / n
  mean_correct = 140.33 := 
by
  sorry

end correct_mean_l38_38285


namespace find_w_l38_38467

variables (w x y z : ℕ)

-- conditions
def condition1 : Prop := x = w / 2
def condition2 : Prop := y = w + x
def condition3 : Prop := z = 400
def condition4 : Prop := w + x + y + z = 1000

-- problem to prove
theorem find_w (h1 : condition1 w x) (h2 : condition2 w x y) (h3 : condition3 z) (h4 : condition4 w x y z) : w = 200 :=
by sorry

end find_w_l38_38467


namespace complex_multiplication_imaginary_unit_l38_38736

theorem complex_multiplication_imaginary_unit 
  (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_imaginary_unit_l38_38736


namespace fraction_value_l38_38113

variable (x y : ℝ)

theorem fraction_value (hx : x = 4) (hy : y = -3) : (x - 2 * y) / (x + y) = 10 := by
  sorry

end fraction_value_l38_38113


namespace k_value_correct_l38_38729

theorem k_value_correct (k : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 + k * x - 8
  (f 5 - g 5 = 20) -> k = 53 / 5 :=
by
  intro h
  sorry

end k_value_correct_l38_38729


namespace max_min_of_f_on_interval_l38_38311

-- Conditions
def f (x : ℝ) : ℝ := x^3 - 3 * x + 1
def interval : Set ℝ := Set.Icc (-3) 0

-- Problem statement
theorem max_min_of_f_on_interval : 
  ∃ (max min : ℝ), max = 1 ∧ min = -17 ∧ 
  (∀ x ∈ interval, f x ≤ max) ∧ 
  (∀ x ∈ interval, f x ≥ min) := 
sorry

end max_min_of_f_on_interval_l38_38311


namespace regular_price_one_pound_is_20_l38_38691

variable (y : ℝ)
variable (discounted_price_quarter_pound : ℝ)

-- Conditions
axiom h1 : 0.6 * (y / 4) + 2 = discounted_price_quarter_pound
axiom h2 : discounted_price_quarter_pound = 2
axiom h3 : 0.1 * y = 2

-- Question: What is the regular price for one pound of cake?
theorem regular_price_one_pound_is_20 : y = 20 := 
  sorry

end regular_price_one_pound_is_20_l38_38691


namespace smallest_nuts_in_bag_l38_38854

theorem smallest_nuts_in_bag :
  ∃ (N : ℕ), N ≡ 1 [MOD 11] ∧ N ≡ 8 [MOD 13] ∧ N ≡ 3 [MOD 17] ∧
             (∀ M, (M ≡ 1 [MOD 11] ∧ M ≡ 8 [MOD 13] ∧ M ≡ 3 [MOD 17]) → M ≥ N) :=
sorry

end smallest_nuts_in_bag_l38_38854


namespace production_line_improvement_better_than_financial_investment_l38_38297

noncomputable def improved_mean_rating (initial_mean : ℝ) := initial_mean + 0.05

noncomputable def combined_mean_rating (mean_unimproved : ℝ) (mean_improved : ℝ) : ℝ :=
  (mean_unimproved * 200 + mean_improved * 200) / 400

noncomputable def combined_variance (variance : ℝ) (combined_mean : ℝ) : ℝ :=
  (2 * variance) - combined_mean ^ 2

noncomputable def increased_returns (grade_a_price : ℝ) (grade_b_price : ℝ) 
  (proportion_upgraded : ℝ) (units_per_day : ℕ) (days_per_year : ℕ) : ℝ :=
  (grade_a_price - grade_b_price) * proportion_upgraded * units_per_day * days_per_year - 200000000

noncomputable def financial_returns (initial_investment : ℝ) (annual_return_rate : ℝ) : ℝ :=
  initial_investment * (1 + annual_return_rate) - initial_investment

theorem production_line_improvement_better_than_financial_investment 
  (initial_mean : ℝ := 9.98) 
  (initial_variance : ℝ := 0.045) 
  (grade_a_price : ℝ := 2000) 
  (grade_b_price : ℝ := 1200) 
  (proportion_upgraded : ℝ := 3 / 8) 
  (units_per_day : ℕ := 200) 
  (days_per_year : ℕ := 365) 
  (initial_investment : ℝ := 200000000) 
  (annual_return_rate : ℝ := 0.082) : 
  combined_mean_rating initial_mean (improved_mean_rating initial_mean) = 10.005 ∧ 
  combined_variance initial_variance (combined_mean_rating initial_mean (improved_mean_rating initial_mean)) = 0.045625 ∧ 
  increased_returns grade_a_price grade_b_price proportion_upgraded units_per_day days_per_year > financial_returns initial_investment annual_return_rate := 
by {
  sorry
}

end production_line_improvement_better_than_financial_investment_l38_38297


namespace stacy_days_to_complete_paper_l38_38696

variable (total_pages pages_per_day : ℕ)
variable (d : ℕ)

theorem stacy_days_to_complete_paper 
  (h1 : total_pages = 63) 
  (h2 : pages_per_day = 21) 
  (h3 : total_pages = pages_per_day * d) : 
  d = 3 := 
sorry

end stacy_days_to_complete_paper_l38_38696


namespace find_a3_plus_a5_l38_38812

-- Define an arithmetic-geometric sequence
def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, 0 < r ∧ ∃ b : ℝ, a n = b * r ^ n

-- Define the given condition
def given_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

-- Define the target theorem statement
theorem find_a3_plus_a5 (a : ℕ → ℝ) 
  (pos_sequence : is_arithmetic_geometric a) 
  (cond : given_condition a) : 
  a 3 + a 5 = 5 :=
sorry

end find_a3_plus_a5_l38_38812


namespace problem1_problem2_l38_38090

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  vector_dot v1 v2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

-- Given vectors in the problem
def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)
def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
def v : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Problem 1: Find k when n is perpendicular to v
theorem problem1 (k : ℝ) : perpendicular (n k) v → k = 5 / 3 := 
by sorry

-- Problem 2: Find k when n is parallel to c + k * b
theorem problem2 (k : ℝ) : parallel (n k) (c.1 + k * b.1, c.2 + k * b.2) → k = -1 / 3 := 
by sorry

end problem1_problem2_l38_38090


namespace polynomial_product_c_l38_38125

theorem polynomial_product_c (b c : ℝ) (h1 : b = 2 * c - 1) (h2 : (x^2 + b * x + c) = 0 → (∃ r : ℝ, x = r)) :
  c = 1 / 2 :=
sorry

end polynomial_product_c_l38_38125


namespace mary_more_than_marco_l38_38423

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l38_38423


namespace TeamC_fee_l38_38500

structure Team :=
(work_rate : ℚ)

def teamA : Team := ⟨1 / 36⟩
def teamB : Team := ⟨1 / 24⟩
def teamC : Team := ⟨1 / 18⟩

def total_fee : ℚ := 36000

def combined_work_rate_first_half (A B C : Team) : ℚ :=
(A.work_rate + B.work_rate + C.work_rate) * 1 / 2

def combined_work_rate_second_half (A C : Team) : ℚ :=
(A.work_rate + C.work_rate) * 1 / 2

def total_work_completed_by_TeamC (A B C : Team) : ℚ :=
C.work_rate * combined_work_rate_first_half A B C + C.work_rate * combined_work_rate_second_half A C

theorem TeamC_fee (A B C : Team) (total_fee : ℚ) :
  total_work_completed_by_TeamC A B C * total_fee = 20000 :=
by
  sorry

end TeamC_fee_l38_38500


namespace circle_equation_l38_38880

theorem circle_equation :
  ∃ M : ℝ × ℝ, (2 * M.1 + M.2 - 1 = 0) ∧
    (∃ r : ℝ, r ≥ 0 ∧ 
      ((3 - M.1)^2 + (0 - M.2)^2 = r^2) ∧
      ((0 - M.1)^2 + (1 - M.2)^2 = r^2)) ∧
    (∃ x y : ℝ, ((x - 1)^2 + (y + 1)^2 = 5)) := 
sorry

end circle_equation_l38_38880


namespace tan_alpha_eq_neg2_complex_expression_eq_neg5_l38_38956

variables (α : ℝ)
variables (h_sin : Real.sin α = - (2 * Real.sqrt 5) / 5)
variables (h_tan_neg : Real.tan α < 0)

theorem tan_alpha_eq_neg2 :
  Real.tan α = -2 :=
sorry

theorem complex_expression_eq_neg5 :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) /
  (Real.cos (α - Real.pi / 2) - Real.sin (3 * Real.pi / 2 + α)) = -5 :=
sorry

end tan_alpha_eq_neg2_complex_expression_eq_neg5_l38_38956


namespace two_pow_a_plus_two_pow_neg_a_l38_38617

theorem two_pow_a_plus_two_pow_neg_a (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end two_pow_a_plus_two_pow_neg_a_l38_38617


namespace intersection_M_N_l38_38207

def set_M : Set ℝ := { x | x * (x - 1) ≤ 0 }
def set_N : Set ℝ := { x | x < 1 }

theorem intersection_M_N : set_M ∩ set_N = { x | 0 ≤ x ∧ x < 1 } := sorry

end intersection_M_N_l38_38207


namespace distinct_configurations_l38_38911

/-- 
Define m, n, and the binomial coefficient function.
conditions:
  - integer grid dimensions m and n with m >= 1, n >= 1.
  - initially (m-1)(n-1) coins in the subgrid of size (m-1) x (n-1).
  - legal move conditions for coins.
question:
  - Prove the number of distinct configurations of coins equals the binomial coefficient.
-/
def number_of_distinct_configurations (m n : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

theorem distinct_configurations (m n : ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) :
  number_of_distinct_configurations m n = Nat.choose (m + n - 2) (m - 1) :=
sorry

end distinct_configurations_l38_38911


namespace distance_greater_than_school_l38_38501

-- Let d1, d2, and d3 be the distances given as the conditions
def distance_orchard_to_house : ℕ := 800
def distance_house_to_pharmacy : ℕ := 1300
def distance_pharmacy_to_school : ℕ := 1700

-- The total distance from orchard to pharmacy via the house
def total_distance_orchard_to_pharmacy : ℕ :=
  distance_orchard_to_house + distance_house_to_pharmacy

-- The difference between the total distance from orchard to pharmacy and the distance from pharmacy to school
def distance_difference : ℕ :=
  total_distance_orchard_to_pharmacy - distance_pharmacy_to_school

-- The theorem to prove
theorem distance_greater_than_school :
  distance_difference = 400 := sorry

end distance_greater_than_school_l38_38501


namespace find_x_value_l38_38775

theorem find_x_value :
  ∃ (x : ℤ), ∀ (y z w : ℤ), (x = 2 * y + 4) → (y = z + 5) → (z = 2 * w + 3) → (w = 50) → x = 220 :=
by
  sorry

end find_x_value_l38_38775


namespace find_a2_given_conditions_l38_38571

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem find_a2_given_conditions
  {a : ℕ → ℤ}
  (h_seq : is_arithmetic_sequence a)
  (h1 : a 3 + a 5 = 24)
  (h2 : a 7 - a 3 = 24) :
  a 2 = 0 :=
by
  sorry

end find_a2_given_conditions_l38_38571


namespace arithmetic_sequence_common_difference_l38_38900

variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_common_difference
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
  (h_non_zero : d ≠ 0)
  (h_sum : a_n 1 + a_n 2 + a_n 3 = 9)
  (h_geom : a_n 2 ^ 2 = a_n 1 * a_n 5) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l38_38900


namespace inequality_holds_l38_38337

theorem inequality_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : a^2 + b^2 ≥ 2 :=
sorry

end inequality_holds_l38_38337


namespace no_three_times_age_ago_l38_38416

theorem no_three_times_age_ago (F D : ℕ) (h₁ : F = 40) (h₂ : D = 40) (h₃ : F = 2 * D) :
  ¬ ∃ x, F - x = 3 * (D - x) :=
by
  sorry

end no_three_times_age_ago_l38_38416


namespace garden_area_l38_38505

variable (L W A : ℕ)
variable (H1 : 3000 = 50 * L)
variable (H2 : 3000 = 15 * (2*L + 2*W))

theorem garden_area : A = 2400 :=
by
  sorry

end garden_area_l38_38505


namespace prime_triples_l38_38037

open Nat

theorem prime_triples (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) :
    (p ∣ q^r + 1) → (q ∣ r^p + 1) → (r ∣ p^q + 1) → (p, q, r) = (2, 5, 3) ∨ (p, q, r) = (3, 2, 5) ∨ (p, q, r) = (5, 3, 2) :=
  by
  sorry

end prime_triples_l38_38037


namespace sheets_per_class_per_day_l38_38129

theorem sheets_per_class_per_day
  (weekly_sheets : ℕ)
  (school_days_per_week : ℕ)
  (num_classes : ℕ)
  (h1 : weekly_sheets = 9000)
  (h2 : school_days_per_week = 5)
  (h3 : num_classes = 9) :
  (weekly_sheets / school_days_per_week) / num_classes = 200 :=
by
  sorry

end sheets_per_class_per_day_l38_38129


namespace problem1_problem2_problem3_l38_38435

-- Proof Problem 1
theorem problem1 : -12 - (-18) + (-7) = -1 := 
by {
  sorry
}

-- Proof Problem 2
theorem problem2 : ((4 / 7) - (1 / 9) + (2 / 21)) * (-63) = -35 := 
by {
  sorry
}

-- Proof Problem 3
theorem problem3 : ((-4) ^ 2) / 2 + 9 * (-1 / 3) - abs (3 - 4) = 4 := 
by {
  sorry
}

end problem1_problem2_problem3_l38_38435


namespace units_digit_of_product_l38_38330

theorem units_digit_of_product (a b c : ℕ) (n m p : ℕ) (units_a : a ≡ 4 [MOD 10])
  (units_b : b ≡ 9 [MOD 10]) (units_c : c ≡ 16 [MOD 10])
  (exp_a : n = 150) (exp_b : m = 151) (exp_c : p = 152) :
  (a^n * b^m * c^p) % 10 = 4 :=
by
  sorry

end units_digit_of_product_l38_38330


namespace right_triangle_perimeter_l38_38654

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (perimeter : ℝ)
  (h1 : area = 180) 
  (h2 : leg1 = 30) 
  (h3 : (1 / 2) * leg1 * leg2 = area)
  (h4 : hypotenuse^2 = leg1^2 + leg2^2)
  (h5 : leg2 = 12) 
  (h6 : hypotenuse = 2 * Real.sqrt 261) :
  perimeter = 42 + 2 * Real.sqrt 261 :=
by
  sorry

end right_triangle_perimeter_l38_38654


namespace find_x_for_equation_l38_38582

theorem find_x_for_equation :
  ∃ x : ℝ, 9 - 3 / (1 / x) + 3 = 3 ↔ x = 3 := 
by 
  sorry

end find_x_for_equation_l38_38582


namespace conditional_probability_chinese_fail_l38_38298

theorem conditional_probability_chinese_fail :
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  P_both / P_chinese = (4 / 7) := by
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  sorry

end conditional_probability_chinese_fail_l38_38298


namespace divisibility_of_polynomial_l38_38273

theorem divisibility_of_polynomial (n : ℕ) (h : n ≥ 1) : 
  ∃ primes : Finset ℕ, primes.card = n ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1) :=
sorry

end divisibility_of_polynomial_l38_38273


namespace arithmetic_sequence_general_term_and_sum_l38_38507

theorem arithmetic_sequence_general_term_and_sum :
  (∃ (a₁ d : ℤ), a₁ + d = 14 ∧ a₁ + 4 * d = 5 ∧ ∀ n : ℤ, a_n = a₁ + (n - 1) * d ∧ (∀ N : ℤ, N ≥ 1 → S_N = N * ((2 * a₁ + (N - 1) * d) / 2) ∧ N = 10 → S_N = 35)) :=
sorry

end arithmetic_sequence_general_term_and_sum_l38_38507


namespace tens_digit_11_pow_2045_l38_38805

theorem tens_digit_11_pow_2045 : 
    ((11 ^ 2045) % 100) / 10 % 10 = 5 :=
by
    sorry

end tens_digit_11_pow_2045_l38_38805


namespace sum_of_vertices_l38_38983

theorem sum_of_vertices (rect_verts: Nat) (pent_verts: Nat) (h1: rect_verts = 4) (h2: pent_verts = 5) : rect_verts + pent_verts = 9 :=
by
  sorry

end sum_of_vertices_l38_38983


namespace routes_from_A_to_B_in_4_by_3_grid_l38_38877

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid_l38_38877


namespace find_a_l38_38413

open Complex

theorem find_a (a : ℝ) (h : (2 + Complex.I * a) / (1 + Complex.I * Real.sqrt 2) = -Complex.I * Real.sqrt 2) :
  a = Real.sqrt 2 := by
  sorry

end find_a_l38_38413


namespace combined_annual_income_after_expenses_l38_38452

noncomputable def brady_monthly_incomes : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
noncomputable def dwayne_monthly_incomes : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_annual_expense : ℕ := 450
def dwayne_annual_expense : ℕ := 300

def annual_income (monthly_incomes : List ℕ) : ℕ :=
  monthly_incomes.foldr (· + ·) 0

theorem combined_annual_income_after_expenses :
  (annual_income brady_monthly_incomes - brady_annual_expense) +
  (annual_income dwayne_monthly_incomes - dwayne_annual_expense) = 3930 :=
by
  sorry

end combined_annual_income_after_expenses_l38_38452


namespace ratio_of_red_to_total_l38_38630

def hanna_erasers : Nat := 4
def tanya_total_erasers : Nat := 20

def rachel_erasers (hanna_erasers : Nat) : Nat :=
  hanna_erasers / 2

def tanya_red_erasers (rachel_erasers : Nat) : Nat :=
  2 * (rachel_erasers + 3)

theorem ratio_of_red_to_total (hanna_erasers tanya_total_erasers : Nat)
  (hanna_has_4 : hanna_erasers = 4) 
  (tanya_total_is_20 : tanya_total_erasers = 20) 
  (twice_as_many : hanna_erasers = 2 * (rachel_erasers hanna_erasers)) 
  (three_less_than_half : rachel_erasers hanna_erasers = (1 / 2:Rat) * (tanya_red_erasers (rachel_erasers hanna_erasers)) - 3) :
  (tanya_red_erasers (rachel_erasers hanna_erasers)) / tanya_total_erasers = 1 / 2 := by
  sorry

end ratio_of_red_to_total_l38_38630


namespace infinite_series_sum_eq_two_l38_38554

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l38_38554


namespace factorize_expression_l38_38974

-- The primary goal is to prove that -2xy^2 + 4xy - 2x = -2x(y - 1)^2
theorem factorize_expression (x y : ℝ) : 
  -2 * x * y^2 + 4 * x * y - 2 * x = -2 * x * (y - 1)^2 := 
by 
  sorry

end factorize_expression_l38_38974


namespace common_ratio_geometric_sequence_l38_38624

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_arith : 2 * (1/2 * a 5) = a 3 + a 4) : q = (1 + Real.sqrt 5) / 2 :=
sorry

end common_ratio_geometric_sequence_l38_38624


namespace retail_women_in_LA_l38_38437

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end retail_women_in_LA_l38_38437


namespace cow_count_16_l38_38503

theorem cow_count_16 (D C : ℕ) 
  (h1 : ∃ (L H : ℕ), L = 2 * D + 4 * C ∧ H = D + C ∧ L = 2 * H + 32) : C = 16 :=
by
  obtain ⟨L, H, ⟨hL, hH, hCond⟩⟩ := h1
  sorry

end cow_count_16_l38_38503


namespace intersection_point_l38_38120

variables (g : ℤ → ℤ) (b a : ℤ)
def g_def := ∀ x : ℤ, g x = 4 * x + b
def inv_def := ∀ y : ℤ, g y = -4 → y = a
def point_intersection := ∀ y : ℤ, (g y = -4) → (y = a) → (a = -16 + b)
def solution : ℤ := -4

theorem intersection_point (b a : ℤ) (h₁ : g_def g b) (h₂ : inv_def g a) (h₃ : point_intersection g a b) :
  a = solution :=
  sorry

end intersection_point_l38_38120


namespace maximum_x_plus_y_l38_38728

theorem maximum_x_plus_y (N x y : ℕ) 
  (hN : N = 19 * x + 95 * y) 
  (hp : ∃ k : ℕ, N = k^2) 
  (hN_le : N ≤ 1995) :
  x + y ≤ 86 :=
sorry

end maximum_x_plus_y_l38_38728


namespace negation_necessary_not_sufficient_l38_38354

theorem negation_necessary_not_sufficient (p q : Prop) : 
  ((¬ p) → ¬ (p ∨ q)) := 
sorry

end negation_necessary_not_sufficient_l38_38354


namespace train_length_l38_38322

theorem train_length (v_train : ℝ) (v_man : ℝ) (t : ℝ) (length_train : ℝ)
  (h1 : v_train = 55) (h2 : v_man = 7) (h3 : t = 10.45077684107852) :
  length_train = 180 :=
by
  sorry

end train_length_l38_38322


namespace math_problem_l38_38166

-- Definitions for the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 0
def condition2 (a b c : ℝ) : Prop := |a| > |b| ∧ |b| > |c|

-- Theorem statement
theorem math_problem (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : c > 0 ∧ a < 0 :=
by
  sorry

end math_problem_l38_38166


namespace seunghye_saw_number_l38_38588

theorem seunghye_saw_number (x : ℝ) (h : 10 * x - x = 37.35) : x = 4.15 :=
by
  sorry

end seunghye_saw_number_l38_38588


namespace value_of_x0_l38_38553

noncomputable def f (x : ℝ) : ℝ := x^3

theorem value_of_x0 (x0 : ℝ) (h1 : f x0 = x0^3) (h2 : deriv f x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
by
  sorry

end value_of_x0_l38_38553


namespace product_of_positive_integer_solutions_l38_38388

theorem product_of_positive_integer_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (n^2 - 47 * n + 660 = p) → False :=
by
  -- Placeholder for proof, based on the problem conditions.
  sorry

end product_of_positive_integer_solutions_l38_38388


namespace gcd_m_n_l38_38949

def m := 122^2 + 234^2 + 345^2 + 10
def n := 123^2 + 233^2 + 347^2 + 10

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l38_38949


namespace mass_percentage_O_correct_l38_38905

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_B : ℝ := 10.81
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_H3BO3 : ℝ := (3 * molar_mass_H) + (1 * molar_mass_B) + (3 * molar_mass_O)

noncomputable def mass_percentage_O_in_H3BO3 : ℝ := ((3 * molar_mass_O) / molar_mass_H3BO3) * 100

theorem mass_percentage_O_correct : abs (mass_percentage_O_in_H3BO3 - 77.59) < 0.01 := 
sorry

end mass_percentage_O_correct_l38_38905


namespace sequence_non_zero_l38_38525

theorem sequence_non_zero :
  ∀ n : ℕ, ∃ a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 1 ∧ a n % 2 = 1) → (a (n+2) = 5 * a (n+1) - 3 * a n)) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 0 ∧ a n % 2 = 0) → (a (n+2) = a (n+1) - a n)) ∧
  (a n ≠ 0) :=
by
  sorry

end sequence_non_zero_l38_38525


namespace fraction_identity_l38_38984

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_identity : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end fraction_identity_l38_38984


namespace percentage_increase_l38_38013

-- Conditions
variables (S_final S_initial : ℝ) (P : ℝ)
def conditions := (S_final = 3135) ∧ (S_initial = 3000) ∧
  (S_final = (S_initial + (P/100) * S_initial) - 0.05 * (S_initial + (P/100) * S_initial))

-- Statement of the problem
theorem percentage_increase (S_final S_initial : ℝ) 
  (cond : conditions S_final S_initial P) : P = 10 := by
  sorry

end percentage_increase_l38_38013


namespace movies_in_first_box_l38_38239

theorem movies_in_first_box (x : ℕ) 
  (cost_first : ℕ) (cost_second : ℕ) 
  (num_second : ℕ) (avg_price : ℕ)
  (h_cost_first : cost_first = 2)
  (h_cost_second : cost_second = 5)
  (h_num_second : num_second = 5)
  (h_avg_price : avg_price = 3)
  (h_total_eq : cost_first * x + cost_second * num_second = avg_price * (x + num_second)) :
  x = 5 :=
by
  sorry

end movies_in_first_box_l38_38239


namespace skye_race_l38_38709

noncomputable def first_part_length := 3

theorem skye_race 
  (total_track_length : ℕ := 6)
  (speed_first_part : ℕ := 150)
  (distance_second_part : ℕ := 2)
  (speed_second_part : ℕ := 200)
  (distance_third_part : ℕ := 1)
  (speed_third_part : ℕ := 300)
  (avg_speed : ℕ := 180) :
  first_part_length = 3 :=
  sorry

end skye_race_l38_38709


namespace certain_percentage_l38_38659

variable {x p : ℝ}

theorem certain_percentage (h1 : 0.40 * x = 160) : p * x = 200 ↔ p = 0.5 := 
by
  sorry

end certain_percentage_l38_38659


namespace new_person_weight_l38_38402

theorem new_person_weight (avg_weight_increase : ℝ) (old_weight new_weight : ℝ) (n : ℕ)
    (weight_increase_per_person : avg_weight_increase = 3.5)
    (number_of_persons : n = 8)
    (replaced_person_weight : old_weight = 62) :
    new_weight = 90 :=
by
  sorry

end new_person_weight_l38_38402


namespace min_inv_sum_l38_38138

theorem min_inv_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 2 * a * 1 + b * 2 = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ (1/a) + (1/b) = 4 :=
sorry

end min_inv_sum_l38_38138


namespace automobile_travel_distance_l38_38793

theorem automobile_travel_distance (b s : ℝ) (h1 : s > 0) :
  let rate := (b / 8) / s  -- rate in meters per second
  let rate_km_per_min := rate * (1 / 1000) * 60  -- convert to kilometers per minute
  let time := 5  -- time in minutes
  rate_km_per_min * time = 3 * b / 80 / s := sorry

end automobile_travel_distance_l38_38793


namespace ricardo_coin_difference_l38_38464

theorem ricardo_coin_difference (p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ 2299) :
  (11500 - 4 * p) - (11500 - 4 * (2300 - p)) = 9192 :=
by
  sorry

end ricardo_coin_difference_l38_38464


namespace matches_in_each_matchbook_l38_38652

-- Conditions given in the problem
def one_stamp_worth_matches (s : ℕ) : Prop := s = 12
def tonya_initial_stamps (t : ℕ) : Prop := t = 13
def tonya_final_stamps (t : ℕ) : Prop := t = 3
def jimmy_initial_matchbooks (j : ℕ) : Prop := j = 5

-- Goal: prove M = 24
theorem matches_in_each_matchbook (M : ℕ) (s t_initial t_final j : ℕ) 
  (h1 : one_stamp_worth_matches s) 
  (h2 : tonya_initial_stamps t_initial) 
  (h3 : tonya_final_stamps t_final) 
  (h4 : jimmy_initial_matchbooks j) : M = 24 := by
  sorry

end matches_in_each_matchbook_l38_38652
