import Mathlib

namespace roller_coaster_height_l837_83776

/-- The required height to ride the roller coaster given Alex's current height,
    normal growth rate, additional growth rate from hanging upside down,
    and the required hanging time. -/
theorem roller_coaster_height
  (current_height : ℝ)
  (normal_growth_rate : ℝ)
  (upside_down_growth_rate : ℝ)
  (hanging_time : ℝ)
  (months_per_year : ℕ)
  (h1 : current_height = 48)
  (h2 : normal_growth_rate = 1 / 3)
  (h3 : upside_down_growth_rate = 1 / 12)
  (h4 : hanging_time = 2)
  (h5 : months_per_year = 12) :
  current_height +
  normal_growth_rate * months_per_year +
  upside_down_growth_rate * hanging_time * months_per_year = 54 :=
by sorry

end roller_coaster_height_l837_83776


namespace geometry_propositions_l837_83795

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (¬ ∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → 
    parallel_line_plane n β → 
    parallel_plane_plane α β → 
    parallel_line_line m n) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → 
    perpendicular_line_plane n β → 
    perpendicular_plane_plane α β → 
    perpendicular_line_line m n) ∧ 
  (¬ ∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → 
    parallel_line_line m n → 
    parallel_line_plane n α) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    parallel_plane_plane α β → 
    perpendicular_line_plane m α → 
    parallel_line_plane n β → 
    perpendicular_line_line m n) := by
  sorry

end geometry_propositions_l837_83795


namespace value_of_M_l837_83787

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end value_of_M_l837_83787


namespace quadratic_equation_and_inequality_l837_83782

theorem quadratic_equation_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x - m ≠ 0) ∧
  (∀ x : ℝ, x^2 + x + 1 > 0) := by
  sorry

end quadratic_equation_and_inequality_l837_83782


namespace min_xy_value_l837_83769

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) :
  x * y ≥ 9 := by
sorry

end min_xy_value_l837_83769


namespace largest_integer_less_than_100_with_remainder_5_mod_8_l837_83725

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end largest_integer_less_than_100_with_remainder_5_mod_8_l837_83725


namespace max_a_is_three_l837_83729

/-- The function f(x) = x^3 - ax is monotonically increasing on [1, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → (x^3 - a*x) ≤ (y^3 - a*y)

/-- The maximum value of 'a' for which f(x) = x^3 - ax is monotonically increasing on [1, +∞) is 3 -/
theorem max_a_is_three :
  (∃ a_max : ℝ, a_max = 3 ∧
    (∀ a : ℝ, is_monotone_increasing a → a ≤ a_max) ∧
    is_monotone_increasing a_max) :=
sorry

end max_a_is_three_l837_83729


namespace smallest_double_multiple_of_2016_l837_83737

def consecutive_double (n : ℕ) : ℕ :=
  n * 1001 + n

theorem smallest_double_multiple_of_2016 :
  ∀ A : ℕ, A < 288 → ¬(∃ k : ℕ, consecutive_double A = 2016 * k) ∧
  ∃ k : ℕ, consecutive_double 288 = 2016 * k :=
by sorry

end smallest_double_multiple_of_2016_l837_83737


namespace special_polynomial_zeros_l837_83774

/-- A polynomial of degree 5 with specific properties -/
def SpecialPolynomial (P : ℂ → ℂ) : Prop :=
  ∃ (r s : ℤ) (a b : ℤ),
    (∀ x, P x = x * (x - r) * (x - s) * (x^2 + a*x + b)) ∧
    (∀ x, ∃ (c : ℤ), P x = c)

theorem special_polynomial_zeros (P : ℂ → ℂ) (h : SpecialPolynomial P) :
  P ((1 + Complex.I * Real.sqrt 15) / 2) = 0 ∧
  P ((1 + Complex.I * Real.sqrt 17) / 2) = 0 :=
sorry

end special_polynomial_zeros_l837_83774


namespace sarah_vacation_reading_l837_83799

/-- Given Sarah's reading speed, book characteristics, and available reading time, prove she can read 6 books. -/
theorem sarah_vacation_reading 
  (reading_speed : ℕ) 
  (words_per_page : ℕ) 
  (pages_per_book : ℕ) 
  (reading_hours : ℕ) 
  (h1 : reading_speed = 40)
  (h2 : words_per_page = 100)
  (h3 : pages_per_book = 80)
  (h4 : reading_hours = 20) : 
  (reading_hours * 60) / ((words_per_page * pages_per_book) / reading_speed) = 6 := by
  sorry

#check sarah_vacation_reading

end sarah_vacation_reading_l837_83799


namespace weight_of_barium_fluoride_l837_83770

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of Barium atoms in BaF2 -/
def num_Ba : ℕ := 1

/-- The number of Fluorine atoms in BaF2 -/
def num_F : ℕ := 2

/-- The number of moles of BaF2 -/
def num_moles : ℝ := 3

/-- Theorem: The weight of 3 moles of Barium fluoride (BaF2) is 525.99 grams -/
theorem weight_of_barium_fluoride :
  (num_moles * (num_Ba * atomic_weight_Ba + num_F * atomic_weight_F)) = 525.99 := by
  sorry

end weight_of_barium_fluoride_l837_83770


namespace eleven_divides_four_digit_palindromes_l837_83709

/-- A four-digit palindrome is a number of the form abba where a and b are digits. -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem eleven_divides_four_digit_palindromes :
  ∀ n : ℕ, FourDigitPalindrome n → 11 ∣ n :=
by sorry

end eleven_divides_four_digit_palindromes_l837_83709


namespace seashells_given_theorem_l837_83727

/-- The number of seashells Sam gave to Joan -/
def seashells_given_to_joan (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that the number of seashells Sam gave to Joan
    is the difference between his initial and current number of seashells -/
theorem seashells_given_theorem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given_to_joan initial_seashells current_seashells = 
  initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given_to_joan 35 17  -- Should output 18

end seashells_given_theorem_l837_83727


namespace smallest_sum_five_consecutive_odd_integers_l837_83784

theorem smallest_sum_five_consecutive_odd_integers : 
  ∀ n : ℕ, n ≥ 35 → 
  ∃ k : ℤ, (k % 2 ≠ 0) ∧ 
  (n = k + (k + 2) + (k + 4) + (k + 6) + (k + 8)) ∧
  (∀ m : ℕ, m < 35 → 
    ¬∃ j : ℤ, (j % 2 ≠ 0) ∧ 
    (m = j + (j + 2) + (j + 4) + (j + 6) + (j + 8))) :=
by
  sorry

end smallest_sum_five_consecutive_odd_integers_l837_83784


namespace bus_driver_compensation_l837_83798

/-- Calculates the total compensation for a bus driver based on hours worked and pay rates -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 50) :
  let overtime_rate := regular_rate * (1 + overtime_percentage)
  let overtime_hours := total_hours - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 920 := by
sorry

end bus_driver_compensation_l837_83798


namespace geometric_series_second_term_l837_83707

theorem geometric_series_second_term 
  (r : ℚ) 
  (sum : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : sum = 10) : 
  let a := sum * (1 - r)
  a * r = 15 / 8 := by
sorry

end geometric_series_second_term_l837_83707


namespace billys_age_l837_83757

/-- Given that the sum of Billy's and Joe's ages is 60 and Billy is three times as old as Joe,
    prove that Billy is 45 years old. -/
theorem billys_age (billy joe : ℕ) 
    (sum_condition : billy + joe = 60)
    (age_ratio : billy = 3 * joe) : 
  billy = 45 := by
  sorry

end billys_age_l837_83757


namespace second_movie_length_second_movie_is_one_and_half_hours_l837_83778

/-- Calculates the length of the second movie given Henri's schedule --/
theorem second_movie_length 
  (total_time : ℝ) 
  (first_movie : ℝ) 
  (reading_rate : ℝ) 
  (words_read : ℝ) : ℝ :=
  let reading_time : ℝ := words_read / (reading_rate * 60)
  let second_movie : ℝ := total_time - first_movie - reading_time
  second_movie

/-- Proves that the length of the second movie is 1.5 hours --/
theorem second_movie_is_one_and_half_hours :
  second_movie_length 8 3.5 10 1800 = 1.5 := by
  sorry

end second_movie_length_second_movie_is_one_and_half_hours_l837_83778


namespace cookies_prepared_l837_83777

theorem cookies_prepared (num_guests : ℕ) (cookies_per_guest : ℕ) : 
  num_guests = 10 → cookies_per_guest = 18 → num_guests * cookies_per_guest = 180 := by
  sorry

#check cookies_prepared

end cookies_prepared_l837_83777


namespace quadratic_not_always_two_roots_l837_83786

theorem quadratic_not_always_two_roots :
  ∃ (a b c : ℝ), b - c > a ∧ a ≠ 0 ∧ ¬(∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end quadratic_not_always_two_roots_l837_83786


namespace inequality_proof_l837_83754

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end inequality_proof_l837_83754


namespace overall_loss_calculation_l837_83700

def stock_worth : ℝ := 15000

def profit_percentage : ℝ := 0.1
def loss_percentage : ℝ := 0.05

def profit_stock_ratio : ℝ := 0.2
def loss_stock_ratio : ℝ := 0.8

def profit_amount : ℝ := stock_worth * profit_stock_ratio * profit_percentage
def loss_amount : ℝ := stock_worth * loss_stock_ratio * loss_percentage

def overall_selling_price : ℝ := 
  (stock_worth * profit_stock_ratio * (1 + profit_percentage)) +
  (stock_worth * loss_stock_ratio * (1 - loss_percentage))

def overall_loss : ℝ := stock_worth - overall_selling_price

theorem overall_loss_calculation :
  overall_loss = 300 :=
sorry

end overall_loss_calculation_l837_83700


namespace evaluate_expression_l837_83753

theorem evaluate_expression : -(18 / 3 * 8 - 80 + 4^2 * 2) = 0 := by
  sorry

end evaluate_expression_l837_83753


namespace horner_method_proof_l837_83791

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^7 + 2x^5 + 4x^3 + x -/
def f_coeffs : List ℤ := [3, 0, 2, 0, 4, 0, 1, 0]

theorem horner_method_proof :
  horner f_coeffs 3 = 7158 := by
  sorry

end horner_method_proof_l837_83791


namespace larger_integer_value_l837_83766

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
sorry

end larger_integer_value_l837_83766


namespace shirt_price_is_16_30_l837_83732

/-- Calculates the final price of a shirt given the cost price, profit percentage, discount percentage, tax rate, and packaging fee. -/
def final_shirt_price (cost_price : ℝ) (profit_percentage : ℝ) (discount_percentage : ℝ) (tax_rate : ℝ) (packaging_fee : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + packaging_fee

/-- Theorem stating that the final price of the shirt is $16.30 given the specific conditions. -/
theorem shirt_price_is_16_30 :
  final_shirt_price 20 0.30 0.50 0.10 2 = 16.30 := by
  sorry

end shirt_price_is_16_30_l837_83732


namespace no_triangle_exists_l837_83741

-- Define the triangle
structure Triangle :=
  (a b : ℝ)
  (angleBisector : ℝ)

-- Define the conditions
def triangleConditions (t : Triangle) : Prop :=
  t.a = 12 ∧ t.b = 20 ∧ t.angleBisector = 15

-- Theorem statement
theorem no_triangle_exists :
  ¬ ∃ (t : Triangle), triangleConditions t ∧ 
  ∃ (c : ℝ), c > 0 ∧ 
  (c + t.a > t.b) ∧ (c + t.b > t.a) ∧ (t.a + t.b > c) ∧
  t.angleBisector = Real.sqrt (t.a * t.b * (1 - (c^2 / (t.a + t.b)^2))) :=
sorry

end no_triangle_exists_l837_83741


namespace peters_erasers_l837_83726

/-- Peter's erasers problem -/
theorem peters_erasers (initial_erasers additional_erasers : ℕ) :
  initial_erasers = 8 →
  additional_erasers = 3 →
  initial_erasers + additional_erasers = 11 := by
  sorry

end peters_erasers_l837_83726


namespace quadratic_factorization_sum_l837_83711

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 13*x + 30 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 5*x - 50 = (x + b)*(x - c)) →
  a + b + c = 18 := by
sorry

end quadratic_factorization_sum_l837_83711


namespace sandy_work_hours_l837_83728

/-- Sandy's work problem -/
theorem sandy_work_hours (hourly_rate : ℚ) (friday_hours : ℚ) (saturday_hours : ℚ) (total_earnings : ℚ) :
  hourly_rate = 15 →
  friday_hours = 10 →
  saturday_hours = 6 →
  total_earnings = 450 →
  (total_earnings - (friday_hours + saturday_hours) * hourly_rate) / hourly_rate = 14 :=
by sorry

end sandy_work_hours_l837_83728


namespace quadratic_equation_roots_l837_83762

theorem quadratic_equation_roots (k : ℝ) : 
  let eq := fun x : ℝ => x^2 + (2*k - 1)*x + k^2 - 1
  ∃ x₁ x₂ : ℝ, 
    (eq x₁ = 0 ∧ eq x₂ = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 := by
  sorry

end quadratic_equation_roots_l837_83762


namespace reciprocal_of_sum_l837_83751

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end reciprocal_of_sum_l837_83751


namespace cubic_function_coefficients_l837_83749

/-- Given a cubic function f(x) = ax³ - 4x² + bx - 3, 
    if f(1) = 3 and f(-2) = -47, then a = 4/3 and b = 26/3 -/
theorem cubic_function_coefficients (a b : ℚ) : 
  let f : ℚ → ℚ := λ x => a * x^3 - 4 * x^2 + b * x - 3
  (f 1 = 3 ∧ f (-2) = -47) → a = 4/3 ∧ b = 26/3 := by
  sorry


end cubic_function_coefficients_l837_83749


namespace plant_supplier_earnings_l837_83718

theorem plant_supplier_earnings :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let pot_cost : ℕ := 150
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + pot_cost
  total_earnings - total_expenses = 1145 :=
by
  sorry

#check plant_supplier_earnings

end plant_supplier_earnings_l837_83718


namespace f_odd_and_increasing_l837_83738

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x y : ℝ, 0 < x → x < y → f x < f y)  -- f is monotonically increasing on (0, +∞)
  := by sorry

end f_odd_and_increasing_l837_83738


namespace number_plus_five_equals_six_l837_83792

theorem number_plus_five_equals_six : ∃ x : ℝ, x + 5 = 6 ∧ x = 1 := by
  sorry

end number_plus_five_equals_six_l837_83792


namespace farm_chicken_ratio_l837_83714

/-- Given a farm with chickens, prove the ratio of hens to roosters -/
theorem farm_chicken_ratio (total : ℕ) (hens : ℕ) (X : ℕ) : 
  total = 75 → 
  hens = 67 → 
  hens = X * (total - hens) - 5 → 
  X = 9 := by
sorry

end farm_chicken_ratio_l837_83714


namespace sequence_bounded_above_l837_83720

/-- The sequence {aₙ} defined by the given recurrence relation is bounded above. -/
theorem sequence_bounded_above (α : ℝ) (h_α : α > 1) :
  ∃ (M : ℝ), ∀ (a : ℕ → ℝ), 
    (a 1 ∈ Set.Ioo 0 1) → 
    (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + (a n / n)^α) → 
    (∀ n : ℕ, n ≥ 1 → a n ≤ M) :=
by sorry

end sequence_bounded_above_l837_83720


namespace quadratic_root_sqrt_5_minus_3_l837_83715

theorem quadratic_root_sqrt_5_minus_3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
  (a = 1 ∧ b = 6 ∧ c = 4) := by
  sorry

end quadratic_root_sqrt_5_minus_3_l837_83715


namespace max_value_of_f_l837_83742

noncomputable def f (x a : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + (5/8) * a + 1

theorem max_value_of_f (a : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y a ≤ f x a) →
  (a < 0 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (5/8) * a + 2) ∧
  (0 ≤ a ∧ a ≤ 2 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = a^2 / 4 + (5/8) * a + 2) ∧
  (2 < a → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (13/8) * a + 1) :=
by sorry

end max_value_of_f_l837_83742


namespace prop1_false_prop2_true_prop3_true_prop4_true_l837_83747

-- Define the basic geometric objects
variable (Line Plane : Type)

-- Define the geometric relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)
variable (line_of_intersection : Plane → Plane → Line)

-- Proposition 1 (false)
theorem prop1_false :
  ¬(∀ (p1 p2 p3 : Plane) (l1 l2 : Line),
    line_in_plane l1 p1 → line_in_plane l2 p1 →
    line_parallel_to_plane l1 p2 → line_parallel_to_plane l2 p2 →
    parallel p1 p2) := sorry

-- Proposition 2 (true)
theorem prop2_true :
  ∀ (p1 p2 : Plane) (l : Line),
    line_perpendicular_to_plane l p1 →
    line_in_plane l p2 →
    perpendicular p1 p2 := sorry

-- Proposition 3 (true)
theorem prop3_true :
  ∀ (l1 l2 l3 : Line),
    lines_perpendicular l1 l3 →
    lines_perpendicular l2 l3 →
    lines_perpendicular l1 l2 := sorry

-- Proposition 4 (true)
theorem prop4_true :
  ∀ (p1 p2 : Plane) (l : Line),
    perpendicular p1 p2 →
    line_in_plane l p1 →
    ¬(lines_perpendicular l (line_of_intersection p1 p2)) →
    ¬(line_perpendicular_to_plane l p2) := sorry

end prop1_false_prop2_true_prop3_true_prop4_true_l837_83747


namespace mrs_white_carrot_yield_l837_83712

/-- Calculates the expected carrot yield from a rectangular garden --/
def expected_carrot_yield (length_steps : ℕ) (width_steps : ℕ) (step_size : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_size * (width_steps : ℚ) * step_size * yield_per_sqft

/-- Proves that the expected carrot yield for Mrs. White's garden is 1875 pounds --/
theorem mrs_white_carrot_yield : 
  expected_carrot_yield 18 25 (5/2) (2/3) = 1875 := by
  sorry

end mrs_white_carrot_yield_l837_83712


namespace two_white_balls_probability_l837_83764

/-- The probability of drawing two white balls from a box with white and black balls -/
theorem two_white_balls_probability
  (total_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 7)
  (h3 : black_balls = 8) :
  (white_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1 / 5 :=
by sorry

end two_white_balls_probability_l837_83764


namespace maria_yearly_distance_l837_83733

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  current_steps : ℕ

/-- Represents a postman's walking data for a year --/
structure PostmanYearlyData where
  pedometer : Pedometer
  flips : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculate the total distance walked by a postman in a year --/
def calculate_yearly_distance (data : PostmanYearlyData) : ℕ :=
  sorry

/-- Maria's yearly walking data --/
def maria_data : PostmanYearlyData :=
  { pedometer := { max_steps := 99999, current_steps := 0 },
    flips := 50,
    final_reading := 25000,
    steps_per_mile := 1500 }

theorem maria_yearly_distance :
  calculate_yearly_distance maria_data = 3350 :=
sorry

end maria_yearly_distance_l837_83733


namespace cube_properties_l837_83790

/-- Given a cube with surface area 864 square units, prove its volume and diagonal length -/
theorem cube_properties (s : ℝ) (h : 6 * s^2 = 864) : 
  s^3 = 1728 ∧ s * Real.sqrt 3 = 12 * Real.sqrt 3 := by
  sorry

end cube_properties_l837_83790


namespace special_divisor_property_implies_prime_l837_83756

theorem special_divisor_property_implies_prime (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) :
  Nat.Prime n :=
sorry

end special_divisor_property_implies_prime_l837_83756


namespace ladder_slide_l837_83755

theorem ladder_slide (ladder_length : ℝ) (initial_base : ℝ) (slip_distance : ℝ) :
  ladder_length = 25 →
  initial_base = 7 →
  slip_distance = 4 →
  ∃ (slide_distance : ℝ),
    slide_distance = 8 ∧
    (ladder_length ^ 2 = (initial_base + slide_distance) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_base ^ 2) - slip_distance) ^ 2) :=
by sorry

end ladder_slide_l837_83755


namespace triplet_transformation_theorem_l837_83740

/-- Represents a triplet of integers -/
structure Triplet where
  a : Int
  b : Int
  c : Int

/-- Represents an operation on a triplet -/
inductive Operation
  | IncrementA (k : Int) (i : Fin 3) : Operation
  | DecrementA (k : Int) (i : Fin 3) : Operation
  | IncrementB (k : Int) (i : Fin 3) : Operation
  | DecrementB (k : Int) (i : Fin 3) : Operation
  | IncrementC (k : Int) (i : Fin 3) : Operation
  | DecrementC (k : Int) (i : Fin 3) : Operation

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.IncrementA k i => { t with a := t.a + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementA k i => { t with a := t.a - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementB k i => { t with b := t.b + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementB k i => { t with b := t.b - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementC k i => { t with c := t.c + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementC k i => { t with c := t.c - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }

theorem triplet_transformation_theorem (a b c : Int) (h : Int.gcd a (Int.gcd b c) = 1) :
  ∃ (ops : List Operation), ops.length ≤ 5 ∧
    (ops.foldl applyOperation (Triplet.mk a b c) = Triplet.mk 1 0 0) := by
  sorry

end triplet_transformation_theorem_l837_83740


namespace triangular_array_sum_l837_83781

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 1.5 * f (n - 1)

/-- The triangular array property -/
theorem triangular_array_sum : f 10 = 38.443359375 := by
  sorry

end triangular_array_sum_l837_83781


namespace factorization_equality_l837_83783

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l837_83783


namespace wider_bolt_width_l837_83708

theorem wider_bolt_width (a b : ℕ) (h1 : a = 45) (h2 : b > a) (h3 : Nat.gcd a b = 15) : 
  (∀ c : ℕ, c > a ∧ Nat.gcd a c = 15 → b ≤ c) → b = 60 := by
  sorry

end wider_bolt_width_l837_83708


namespace max_value_quadratic_l837_83767

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 19 ∧ ∀ p : ℝ, -3 * p^2 + 18 * p - 8 ≤ M :=
by sorry

end max_value_quadratic_l837_83767


namespace brown_eyed_brunettes_count_l837_83773

/-- Represents the number of girls with specific characteristics -/
structure GirlCount where
  total : ℕ
  greenEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes given the counts of girls with specific characteristics -/
def brownEyedBrunettes (gc : GirlCount) : ℕ :=
  gc.brownEyed - (gc.total - gc.brunettes - gc.greenEyedBlondes)

/-- Theorem stating that given the specific counts, there are 20 brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (gc : GirlCount) 
  (h1 : gc.total = 60)
  (h2 : gc.greenEyedBlondes = 20)
  (h3 : gc.brunettes = 35)
  (h4 : gc.brownEyed = 25) :
  brownEyedBrunettes gc = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, greenEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end brown_eyed_brunettes_count_l837_83773


namespace percentage_calculation_l837_83793

theorem percentage_calculation (x : ℝ) :
  (30 / 100) * ((60 / 100) * ((70 / 100) * x)) = (126 / 1000) * x := by
  sorry

end percentage_calculation_l837_83793


namespace perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l837_83734

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers, while 48 and 28 are not -/
theorem perfect_number_examples :
  is_perfect_number 29 ∧ is_perfect_number 13 ∧ ¬is_perfect_number 48 ∧ ¬is_perfect_number 28 :=
sorry

/-- Statement 2: Given a^2 - 4a + 8 = (a - m)^2 + n^2, prove that mn = ±4 -/
theorem mn_value (a m n : ℝ) (h : a^2 - 4*a + 8 = (a - m)^2 + n^2) :
  m * n = 4 ∨ m * n = -4 :=
sorry

/-- Statement 3: Given S = a^2 + 4ab + 5b^2 - 12b + k, prove that S is a perfect number when k = 36 -/
theorem S_is_perfect_number (a b : ℤ) :
  is_perfect_number (a^2 + 4*a*b + 5*b^2 - 12*b + 36) :=
sorry

/-- Statement 4: Given -a^2 + 5a + b - 7 = 0, prove that the minimum value of a + b is 3 -/
theorem min_value_a_plus_b (a b : ℝ) (h : -a^2 + 5*a + b - 7 = 0) :
  a + b ≥ 3 :=
sorry

end perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l837_83734


namespace rectangle_dimension_change_l837_83788

theorem rectangle_dimension_change (l b : ℝ) (h1 : l > 0) (h2 : b > 0) : 
  let new_l := l / 2
  let new_area := (l * b) / 2
  ∃ new_b : ℝ, new_area = new_l * new_b ∧ new_b = b / 2 :=
by sorry

end rectangle_dimension_change_l837_83788


namespace discount_comparison_l837_83779

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.10]
def option2_discounts : List ℝ := [0.30, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 135 :=
by sorry

end discount_comparison_l837_83779


namespace brothers_snowballs_l837_83796

theorem brothers_snowballs (janet_snowballs : ℕ) (janet_percentage : ℚ) : 
  janet_snowballs = 50 → janet_percentage = 1/4 → 
  (1 - janet_percentage) * (janet_snowballs / janet_percentage) = 150 := by
  sorry

end brothers_snowballs_l837_83796


namespace integral_exp_plus_x_l837_83752

theorem integral_exp_plus_x : ∫ x in (0 : ℝ)..(1 : ℝ), (Real.exp x + x) = Real.exp 1 - 1 / 2 := by
  sorry

end integral_exp_plus_x_l837_83752


namespace restaurant_bill_calculation_l837_83761

theorem restaurant_bill_calculation :
  let num_bankers : ℕ := 4
  let num_clients : ℕ := 5
  let total_people : ℕ := num_bankers + num_clients
  let cost_per_person : ℚ := 70
  let gratuity_rate : ℚ := 0.20
  let pre_gratuity_total : ℚ := total_people * cost_per_person
  let gratuity_amount : ℚ := pre_gratuity_total * gratuity_rate
  let total_bill : ℚ := pre_gratuity_total + gratuity_amount
  total_bill = 756 := by
sorry

end restaurant_bill_calculation_l837_83761


namespace chocolate_box_bars_l837_83736

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_price : ℚ := 3
  let remaining_bars : ℕ := 4
  let sales : ℚ := 9
  bar_price * (total_bars - remaining_bars) = sales

theorem chocolate_box_bars : ∃ (x : ℕ), chocolate_problem x ∧ x = 7 := by
  sorry

end chocolate_box_bars_l837_83736


namespace parallel_vectors_k_value_l837_83743

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  let a : ℝ × ℝ := (2*k - 3, -6)
  let c : ℝ × ℝ := (2, 1)
  are_parallel a c → k = -9/2 := by
sorry

end parallel_vectors_k_value_l837_83743


namespace percentage_relation_l837_83716

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A = 0.06 * C := by
  sorry

end percentage_relation_l837_83716


namespace perimeter_difference_rectangle_and_squares_l837_83789

/-- The perimeter of a rectangle with given length and width -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a single unit square -/
def unit_square_perimeter : ℕ := 4

/-- The perimeter of n non-overlapping unit squares arranged in a straight line -/
def n_unit_squares_perimeter (n : ℕ) : ℕ := n * unit_square_perimeter

theorem perimeter_difference_rectangle_and_squares : 
  (rectangle_perimeter 6 1) - (n_unit_squares_perimeter 3) = 2 := by
  sorry

end perimeter_difference_rectangle_and_squares_l837_83789


namespace angle_C_measure_l837_83724

/-- Represents a hexagon CALCUL with specific angle properties -/
structure Hexagon where
  -- Angles of the hexagon
  A : ℝ
  C : ℝ
  L : ℝ
  U : ℝ
  -- Conditions
  angle_sum : A + C + L + U + L + C = 720
  C_eq_L_eq_U : C = L ∧ L = U
  A_eq_L_eq_C : A = L ∧ L = C
  A_L_supplementary : A + L = 180

/-- The measure of angle C in the hexagon CALCUL is 120° -/
theorem angle_C_measure (h : Hexagon) : h.C = 120 := by
  sorry

end angle_C_measure_l837_83724


namespace sphere_only_identical_views_l837_83706

-- Define the set of common 3D solids
inductive Solid
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

-- Define a function to check if all views are identical
def allViewsIdentical (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ s : Solid, allViewsIdentical s ↔ s = Solid.Sphere :=
sorry

end sphere_only_identical_views_l837_83706


namespace exp_sum_gt_two_l837_83758

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem exp_sum_gt_two (ha : a ≠ 0) (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) (hf₂ : f a x₂ = 0) : 
  Real.exp (a * x₁) + Real.exp (a * x₂) > 2 :=
by sorry

end

end exp_sum_gt_two_l837_83758


namespace min_value_3x_plus_4y_l837_83748

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
sorry

end min_value_3x_plus_4y_l837_83748


namespace square_nonnegative_l837_83704

theorem square_nonnegative (a : ℝ) : a^2 ≥ 0 := by
  sorry

end square_nonnegative_l837_83704


namespace complex_fraction_evaluation_l837_83703

theorem complex_fraction_evaluation : 
  (((1 / 2) * (1 / 3) * (1 / 4) * (1 / 5) + (3 / 2) * (3 / 4) * (3 / 5)) / 
   ((1 / 2) * (2 / 3) * (2 / 5))) = 41 / 8 := by
  sorry

end complex_fraction_evaluation_l837_83703


namespace square_difference_equals_736_l837_83710

theorem square_difference_equals_736 : (23 + 16)^2 - (23^2 + 16^2) = 736 := by
  sorry

end square_difference_equals_736_l837_83710


namespace min_sum_squares_l837_83780

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min :=
by sorry

end min_sum_squares_l837_83780


namespace complement_of_A_in_U_l837_83750

def U : Set ℕ := {x : ℕ | x > 0 ∧ (x - 6) * (x + 1) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {3, 5, 6} := by sorry

end complement_of_A_in_U_l837_83750


namespace rare_coin_collection_l837_83719

theorem rare_coin_collection (initial_gold : ℕ) (initial_silver : ℕ) : 
  initial_gold = initial_silver / 3 →
  (initial_gold + 15 : ℕ) = initial_silver / 2 →
  initial_gold + initial_silver + 15 = 135 :=
by sorry

end rare_coin_collection_l837_83719


namespace checkerboard_sums_l837_83785

/-- Represents a 10x10 checkerboard filled with numbers 1 to 100 -/
def Checkerboard := Fin 10 → Fin 10 → Nat

/-- The checkerboard filled with numbers 1 to 100 in order -/
def filledCheckerboard : Checkerboard :=
  fun i j => i.val * 10 + j.val + 1

/-- The sum of the corner numbers on the checkerboard -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 9 + board 9 0 + board 9 9

/-- The sum of the main diagonal numbers on the checkerboard -/
def diagonalSum (board : Checkerboard) : Nat :=
  board 0 0 + board 9 9

theorem checkerboard_sums :
  cornerSum filledCheckerboard = 202 ∧
  diagonalSum filledCheckerboard = 101 := by
  sorry


end checkerboard_sums_l837_83785


namespace lcm_from_product_and_hcf_l837_83765

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end lcm_from_product_and_hcf_l837_83765


namespace square_fraction_count_l837_83745

theorem square_fraction_count : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 29 ∧ ∃ k : ℤ, n / (30 - n) = k^2) ∧ 
    (∀ n : ℤ, 0 ≤ n ∧ n ≤ 29 ∧ (∃ k : ℤ, n / (30 - n) = k^2) → n ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end square_fraction_count_l837_83745


namespace cookie_problem_l837_83744

theorem cookie_problem :
  let n : ℕ := 1817
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(m % 6 = 5 ∧ m % 7 = 3 ∧ m % 9 = 7 ∧ m % 11 = 10)) ∧
  (n % 6 = 5 ∧ n % 7 = 3 ∧ n % 9 = 7 ∧ n % 11 = 10) := by
  sorry

end cookie_problem_l837_83744


namespace min_value_expression_l837_83705

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 := by
  sorry

end min_value_expression_l837_83705


namespace min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l837_83739

-- Define the curve M and line l
def curve_M (x y : ℝ) : Prop := (x^2 = -y) ∨ (x^2 = 4*y)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3

-- Define the number of intersection points
def intersection_points (k : ℝ) : ℕ := sorry

-- Theorem 1: Minimum value of k when m ≥ 3
theorem min_k_for_three_or_more_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k ≥ 3 → k ≥ Real.sqrt 3 := by sorry

-- Theorem 2: Range of |AB|/|CD| when m = 4
theorem range_of_ratio_for_four_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k = 4 →
  ∃ r : ℝ, 0 < r ∧ r < 4 ∧
  (∃ A B C D : ℝ × ℝ,
    curve_M A.1 A.2 ∧ curve_M B.1 B.2 ∧ curve_M C.1 C.2 ∧ curve_M D.1 D.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k C.1 C.2 ∧ line_l k D.1 D.2 ∧
    r = (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) /
        (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2))) := by sorry

end min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l837_83739


namespace intersection_singleton_l837_83797

/-- The set A defined by the equation y = ax + 1 -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1 + 1}

/-- The set B defined by the equation y = |x| -/
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = |p.1|}

/-- Theorem stating the condition for A ∩ B to be a singleton set -/
theorem intersection_singleton (a : ℝ) : (A a ∩ B).Finite ∧ (A a ∩ B).Nonempty ↔ a ≥ 1 ∨ a ≤ -1 := by
  sorry

end intersection_singleton_l837_83797


namespace sum_of_min_max_x_l837_83768

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 10) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 10 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 16/3 := by
  sorry

end sum_of_min_max_x_l837_83768


namespace liquid_X_percentage_l837_83731

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 0.0009

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 0.018

/-- The weight of solution A in grams -/
def weight_A : ℝ := 200

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.0142

theorem liquid_X_percentage :
  percentage_X_in_A * weight_A + percentage_X_in_B * weight_B =
  percentage_X_in_mixture * (weight_A + weight_B) := by
  sorry

end liquid_X_percentage_l837_83731


namespace new_songs_added_l837_83730

def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

theorem new_songs_added : 
  final_songs - (initial_songs - deleted_songs) = 20 := by sorry

end new_songs_added_l837_83730


namespace upper_limit_of_set_W_l837_83723

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_W (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | n > 10 ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_W (upper_bound : ℕ) :
  (∃ (w : Set ℕ), w = set_W upper_bound ∧ 
   (∃ (max min : ℕ), max ∈ w ∧ min ∈ w ∧ 
    (∀ x ∈ w, x ≤ max ∧ x ≥ min) ∧ max - min = 12)) →
  upper_bound = 23 :=
sorry

end upper_limit_of_set_W_l837_83723


namespace hyperbola_eccentricity_l837_83721

/-- The hyperbola struct represents a hyperbola with semi-major axis a and semi-minor axis b. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The Point struct represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The Focus struct represents a focus point of the hyperbola. -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Represents the condition that P is on the right branch of the hyperbola. -/
def is_on_right_branch (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1 ∧ p.x > 0

/-- Represents the condition that the distance from O to PF₁ equals the real semi-axis. -/
def distance_condition (h : Hyperbola) (p : Point) (f₁ : Focus) : Prop :=
  ∃ (d : ℝ), d = h.a ∧ d = abs (f₁.y * p.x - f₁.x * p.y) / Real.sqrt ((p.x - f₁.x)^2 + (p.y - f₁.y)^2)

/-- The main theorem stating the eccentricity of the hyperbola under given conditions. -/
theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) (f₁ f₂ : Focus) :
  is_on_right_branch h p →
  (p.x - f₂.x)^2 + (p.y - f₂.y)^2 = (f₁.x - f₂.x)^2 + (f₁.y - f₂.y)^2 →
  distance_condition h p f₁ →
  let e := Real.sqrt (1 + h.b^2 / h.a^2)
  e = 5 / 3 := by
  sorry

end hyperbola_eccentricity_l837_83721


namespace cody_candy_count_l837_83760

/-- The number of boxes of chocolate candy Cody bought -/
def chocolate_boxes : ℕ := 7

/-- The number of boxes of caramel candy Cody bought -/
def caramel_boxes : ℕ := 3

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 8

/-- The total number of candy pieces Cody has -/
def total_candy : ℕ := (chocolate_boxes + caramel_boxes) * pieces_per_box

theorem cody_candy_count : total_candy = 80 := by
  sorry

end cody_candy_count_l837_83760


namespace sum_of_squares_zero_implies_sum_l837_83759

theorem sum_of_squares_zero_implies_sum (a b c d : ℝ) :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 →
  a + b + c + d = 16 := by
  sorry

end sum_of_squares_zero_implies_sum_l837_83759


namespace equation_equality_l837_83702

theorem equation_equality (x : ℝ) : 
  4 * x^4 + x^3 - 2*x + 5 + (-4 * x^4 + x^3 - 7 * x^2 + 2*x - 1) = 2 * x^3 - 7 * x^2 + 4 := by
  sorry

end equation_equality_l837_83702


namespace apples_eaten_proof_l837_83771

-- Define the daily apple consumption for each person
def simone_daily : ℚ := 1/2
def lauri_daily : ℚ := 1/3
def alex_daily : ℚ := 1/4

-- Define the number of days each person ate apples
def simone_days : ℕ := 16
def lauri_days : ℕ := 15
def alex_days : ℕ := 20

-- Define the total number of apples eaten by all three
def total_apples : ℚ := simone_daily * simone_days + lauri_daily * lauri_days + alex_daily * alex_days

-- Theorem statement
theorem apples_eaten_proof : total_apples = 18 := by
  sorry

end apples_eaten_proof_l837_83771


namespace fifteen_percent_of_600_is_90_l837_83713

theorem fifteen_percent_of_600_is_90 : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end fifteen_percent_of_600_is_90_l837_83713


namespace certain_number_problem_l837_83746

theorem certain_number_problem (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end certain_number_problem_l837_83746


namespace special_number_theorem_l837_83794

def is_nine_digit_number (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n ≤ 999999999

def has_special_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a ∧
    b = 2 * a

def is_product_of_five_primes_squared (n : ℕ) : Prop :=
  ∃ (p q r s t : ℕ),
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
    r ≠ s ∧ r ≠ t ∧
    s ≠ t ∧
    n = (p * q * r * s * t) ^ 2

theorem special_number_theorem (n : ℕ) :
  is_nine_digit_number n ∧
  has_special_form n ∧
  satisfies_condition n ∧
  is_product_of_five_primes_squared n →
  n = 100200100 ∨ n = 225450225 := by
sorry

end special_number_theorem_l837_83794


namespace square_area_error_percentage_l837_83735

/-- If the measured side length of a square is 102.5% of its actual side length,
    then the percentage of error in the calculated area of the square is 5.0625%. -/
theorem square_area_error_percentage (a : ℝ) (h : a > 0) :
  let measured_side := 1.025 * a
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  (calculated_area - actual_area) / actual_area * 100 = 5.0625 := by
sorry

end square_area_error_percentage_l837_83735


namespace polygon_sides_count_l837_83717

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The theorem stating that a polygon satisfying the given condition has 9 sides -/
theorem polygon_sides_count : 
  ∃ (n : ℕ), n > 0 ∧ 
  (num_diagonals (2*n) - 2*n) - (num_diagonals n - n) = 99 ∧ 
  n = 9 := by
  sorry

end polygon_sides_count_l837_83717


namespace ad_greater_bc_l837_83701

theorem ad_greater_bc (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end ad_greater_bc_l837_83701


namespace paper_cup_cost_theorem_l837_83775

/-- The total number of pallets -/
def total_pallets : ℕ := 20

/-- The number of paper towel pallets -/
def paper_towel_pallets : ℕ := total_pallets / 2

/-- The number of tissue pallets -/
def tissue_pallets : ℕ := total_pallets / 4

/-- The number of paper plate pallets -/
def paper_plate_pallets : ℕ := total_pallets / 5

/-- The number of paper cup pallets -/
def paper_cup_pallets : ℕ := total_pallets - (paper_towel_pallets + tissue_pallets + paper_plate_pallets)

/-- The cost of a single paper cup pallet -/
def paper_cup_pallet_cost : ℕ := 50

/-- The total cost spent on paper cup pallets -/
def total_paper_cup_cost : ℕ := paper_cup_pallets * paper_cup_pallet_cost

theorem paper_cup_cost_theorem : total_paper_cup_cost = 50 := by
  sorry

end paper_cup_cost_theorem_l837_83775


namespace cubic_symmetry_about_origin_l837_83763

def f (x : ℝ) : ℝ := x^3

theorem cubic_symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -f x :=
by sorry

end cubic_symmetry_about_origin_l837_83763


namespace only_one_equals_sum_of_squares_of_digits_l837_83772

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The only positive integer n such that s(n) = n is 1 -/
theorem only_one_equals_sum_of_squares_of_digits :
  ∀ n : ℕ, n > 0 → (sum_of_squares_of_digits n = n ↔ n = 1) :=
by sorry

end only_one_equals_sum_of_squares_of_digits_l837_83772


namespace area_of_PQRSUV_l837_83722

-- Define the polygon and its components
structure Polygon where
  PQ : ℝ
  QR : ℝ
  UV : ℝ
  SU : ℝ
  TU : ℝ
  RS : ℝ

-- Define the conditions
def polygon_conditions (p : Polygon) : Prop :=
  p.PQ = 8 ∧
  p.QR = 10 ∧
  p.UV = 6 ∧
  p.SU = 3 ∧
  p.PQ = p.TU + p.UV ∧
  p.QR = p.RS + p.SU

-- Define the area calculation
def area_PQRSUV (p : Polygon) : ℝ :=
  p.PQ * p.QR - p.SU * p.UV

-- Theorem statement
theorem area_of_PQRSUV (p : Polygon) (h : polygon_conditions p) :
  area_PQRSUV p = 62 := by
  sorry

end area_of_PQRSUV_l837_83722
