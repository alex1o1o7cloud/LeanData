import Mathlib

namespace NUMINAMATH_CALUDE_project_time_ratio_l3843_384372

theorem project_time_ratio (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  2 * kate_hours + kate_hours + (kate_hours + 85) = total_hours →
  (2 * kate_hours) / (kate_hours + 85) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_project_time_ratio_l3843_384372


namespace NUMINAMATH_CALUDE_jerry_fifth_night_earnings_l3843_384371

def jerry_tips (n : ℕ) : List ℝ := [20, 60, 15, 40]
def days_worked : ℕ := 5
def target_average : ℝ := 50

theorem jerry_fifth_night_earnings :
  let total_target : ℝ := days_worked * target_average
  let current_total : ℝ := (jerry_tips days_worked).sum
  let required_fifth_night : ℝ := total_target - current_total
  required_fifth_night = 115 := by sorry

end NUMINAMATH_CALUDE_jerry_fifth_night_earnings_l3843_384371


namespace NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l3843_384367

theorem greatest_integer_for_integer_fraction : 
  (∀ y : ℤ, y > 35 → ¬(∃ n : ℤ, (y^2 + 2*y + 7) / (y - 4) = n)) ∧ 
  (∃ n : ℤ, (35^2 + 2*35 + 7) / (35 - 4) = n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l3843_384367


namespace NUMINAMATH_CALUDE_calculate_income_l3843_384369

/-- Represents a person's monthly income and expenses -/
structure MonthlyFinances where
  income : ℝ
  household_percent : ℝ
  clothes_percent : ℝ
  medicines_percent : ℝ
  savings : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem calculate_income (finances : MonthlyFinances)
  (household_cond : finances.household_percent = 35)
  (clothes_cond : finances.clothes_percent = 20)
  (medicines_cond : finances.medicines_percent = 5)
  (savings_cond : finances.savings = 15000)
  (total_cond : finances.household_percent + finances.clothes_percent + finances.medicines_percent + (finances.savings / finances.income * 100) = 100) :
  finances.income = 37500 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_l3843_384369


namespace NUMINAMATH_CALUDE_prob_green_face_specific_die_l3843_384366

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  total_faces_eq : sides = red_faces + green_faces + blue_faces

/-- The probability of rolling a green face on a colored die -/
def prob_green_face (d : ColoredDie) : ℚ :=
  d.green_faces / d.sides

/-- Theorem: The probability of rolling a green face on a 10-sided die
    with 5 red faces, 3 green faces, and 2 blue faces is 3/10 -/
theorem prob_green_face_specific_die :
  let d : ColoredDie := {
    sides := 10,
    red_faces := 5,
    green_faces := 3,
    blue_faces := 2,
    total_faces_eq := by rfl
  }
  prob_green_face d = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_specific_die_l3843_384366


namespace NUMINAMATH_CALUDE_solar_project_analysis_l3843_384342

/-- Represents the net profit of a solar power generation project over n years -/
def net_profit (n : ℕ+) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the average annual profit of the project over n years -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem solar_project_analysis :
  ∀ n : ℕ+,
  -- 1. Net profit function
  net_profit n = -4 * n^2 + 80 * n - 144 ∧
  -- 2. Project starts making profit from the 3rd year
  (∀ k : ℕ+, k ≥ 3 → net_profit k > 0) ∧
  (∀ k : ℕ+, k < 3 → net_profit k ≤ 0) ∧
  -- 3. Maximum average annual profit occurs when n = 6
  (∀ k : ℕ+, avg_annual_profit k ≤ avg_annual_profit 6) ∧
  -- 4. Maximum net profit occurs when n = 10
  (∀ k : ℕ+, net_profit k ≤ net_profit 10) ∧
  -- 5. Both options result in the same total profit
  net_profit 6 + 72 = net_profit 10 + 8 ∧
  net_profit 6 + 72 = 264 :=
by sorry


end NUMINAMATH_CALUDE_solar_project_analysis_l3843_384342


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3843_384334

-- Define repeating decimals
def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_27 : ℚ := 3 / 11

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_3 + repeating_decimal_27 = 20 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3843_384334


namespace NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l3843_384341

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in Set.Icc (Real.exp 1) b, 2 / x) = 6 →
  b = Real.exp 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l3843_384341


namespace NUMINAMATH_CALUDE_supermarket_turnover_equation_l3843_384321

/-- Represents the equation for a supermarket's quarterly turnover with monthly growth rate -/
theorem supermarket_turnover_equation (x : ℝ) : 
  200 * (1 + (1 + x) + (1 + x)^2) = 1000 ↔ 
  (2 * (1 + x + (1 + x)^2) = 10 ∧ 
   2 > 0 ∧ 
   10 > 0 ∧ 
   (∀ m : ℕ, m < 3 → (1 + x)^m > 0)) := by
  sorry

end NUMINAMATH_CALUDE_supermarket_turnover_equation_l3843_384321


namespace NUMINAMATH_CALUDE_problem_solution_l3843_384344

theorem problem_solution (x y : ℝ) : 
  x > 0 → 
  y > 0 → 
  x / 100 * y = 5 → 
  y = 2 * x + 10 → 
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3843_384344


namespace NUMINAMATH_CALUDE_worker_completion_time_proof_l3843_384315

/-- Represents a worker with their working days and payment --/
structure Worker where
  days : ℕ
  payment : ℕ

/-- Calculates the time it would take a worker to complete the entire job --/
def timeToCompleteJob (w : Worker) (totalPayment : ℕ) : ℕ :=
  w.days * (totalPayment / w.payment)

theorem worker_completion_time_proof (w1 w2 w3 : Worker) 
  (h1 : w1.days = 6 ∧ w1.payment = 36)
  (h2 : w2.days = 3 ∧ w2.payment = 12)
  (h3 : w3.days = 8 ∧ w3.payment = 24) :
  let totalPayment := w1.payment + w2.payment + w3.payment
  (timeToCompleteJob w1 totalPayment = 12) ∧
  (timeToCompleteJob w2 totalPayment = 18) ∧
  (timeToCompleteJob w3 totalPayment = 24) := by
  sorry

#check worker_completion_time_proof

end NUMINAMATH_CALUDE_worker_completion_time_proof_l3843_384315


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3843_384355

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  (4 * Real.pi * r^2 : ℝ) = 36 * 2^(2/3) * Real.pi ↔ (4/3 * Real.pi * r^3 : ℝ) = V := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l3843_384355


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3843_384391

/-- The area of the union of a square and an inscribed circle -/
theorem square_circle_union_area 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 20) 
  (h2 : circle_radius = 10) 
  (h3 : circle_radius = square_side / 2) : 
  square_side ^ 2 = 400 := by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3843_384391


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3843_384376

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m passing through point (x₀, y₀) -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Theorem about an ellipse with specific properties and its intersection with a line -/
theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.a^2 = 12 ∧ C.b = 2 ∧ (C.a^2 - C.b^2 = 8) ∧ 
  l.m = 1 ∧ l.x₀ = -2 ∧ l.y₀ = 1 →
  (∃ A B : ℝ × ℝ,
    (A.1^2 / 12 + A.2^2 / 4 = 1) ∧
    (B.1^2 / 12 + B.2^2 / 4 = 1) ∧
    (A.2 = A.1 + 3) ∧
    (B.2 = B.1 + 3) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 42 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3843_384376


namespace NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3843_384375

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x < 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3843_384375


namespace NUMINAMATH_CALUDE_average_marks_is_76_l3843_384348

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def biology_marks : ℕ := 82

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_is_76 : (total_marks : ℚ) / num_subjects = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_is_76_l3843_384348


namespace NUMINAMATH_CALUDE_store_marbles_proof_l3843_384368

/-- The number of marbles initially in the store, given the number of customers,
    marbles bought per customer, and remaining marbles after sales. -/
def initial_marbles (customers : ℕ) (marbles_per_customer : ℕ) (remaining_marbles : ℕ) : ℕ :=
  customers * marbles_per_customer + remaining_marbles

theorem store_marbles_proof :
  initial_marbles 20 15 100 = 400 :=
by sorry

end NUMINAMATH_CALUDE_store_marbles_proof_l3843_384368


namespace NUMINAMATH_CALUDE_store_savings_l3843_384327

/-- The difference between the selling price and the store's cost for a pair of pants. -/
def price_difference (selling_price store_cost : ℕ) : ℕ :=
  selling_price - store_cost

/-- Theorem stating that the price difference is 8 dollars given the specific selling price and store cost. -/
theorem store_savings : price_difference 34 26 = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_savings_l3843_384327


namespace NUMINAMATH_CALUDE_job_completion_time_l3843_384308

/-- Given that:
    - A can do a job in 45 days
    - A and B working together can finish 4 times the amount of work in 72 days
    Prove that B can do the job alone in 30 days -/
theorem job_completion_time (a_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) (b_time : ℝ) :
  a_time = 45 →
  combined_time = 72 →
  combined_work = 4 →
  (1 / a_time + 1 / b_time) * combined_time = combined_work →
  b_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3843_384308


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3843_384395

theorem shaded_area_percentage (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 4)
  (h3 : large_side > 0)
  (h4 : small_side > 0)
  (h5 : small_side < large_side) :
  (large_side^2 - small_side^2) / large_side^2 * 100 = 84 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3843_384395


namespace NUMINAMATH_CALUDE_triangle_median_inequalities_l3843_384304

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, prove two inequalities involving the medians. -/
theorem triangle_median_inequalities (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 / (b*c) + mb^2 / (c*a) + mc^2 / (a*b) ≥ 9/4) ∧
  ((mb^2 + mc^2 - ma^2) / (b*c) + (mc^2 + ma^2 - mb^2) / (c*a) + (ma^2 + mb^2 - mc^2) / (a*b) ≥ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_inequalities_l3843_384304


namespace NUMINAMATH_CALUDE_correct_adult_ticket_cost_l3843_384337

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The number of adults in the family -/
def num_adults : ℕ := 2

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The total cost of tickets for the family -/
def total_cost : ℕ := 77

theorem correct_adult_ticket_cost :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_adult_ticket_cost_l3843_384337


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3843_384309

/-- Given the relationships between Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem sum_of_numbers (mickey jayden coraline : ℕ) 
    (h1 : mickey = jayden + 20)
    (h2 : jayden = coraline - 40)
    (h3 : coraline = 80) : 
  mickey + jayden + coraline = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3843_384309


namespace NUMINAMATH_CALUDE_fraction_equality_l3843_384361

theorem fraction_equality : (20 * 2 + 10) / (5 + 3 - 1) = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3843_384361


namespace NUMINAMATH_CALUDE_james_partner_teaching_difference_l3843_384382

/-- Proves that the difference in teaching years between James and his partner is 10 -/
theorem james_partner_teaching_difference :
  ∀ (james_years partner_years : ℕ),
    james_years = 40 →
    james_years + partner_years = 70 →
    partner_years < james_years →
    james_years - partner_years = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_partner_teaching_difference_l3843_384382


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3843_384381

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3843_384381


namespace NUMINAMATH_CALUDE_female_fraction_is_four_fifths_l3843_384328

/-- Represents a corporation with male and female employees -/
structure Corporation where
  maleEmployees : ℕ
  femaleEmployees : ℕ

/-- The fraction of employees who are at least 35 years old -/
def atLeast35Fraction (c : Corporation) : ℚ :=
  (0.5 * c.maleEmployees + 0.4 * c.femaleEmployees) / (c.maleEmployees + c.femaleEmployees)

/-- The fraction of employees who are females -/
def femaleFraction (c : Corporation) : ℚ :=
  c.femaleEmployees / (c.maleEmployees + c.femaleEmployees)

theorem female_fraction_is_four_fifths (c : Corporation) 
    (h : atLeast35Fraction c = 0.42) : 
    femaleFraction c = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_female_fraction_is_four_fifths_l3843_384328


namespace NUMINAMATH_CALUDE_max_product_of_functions_l3843_384335

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_of_functions (h k : ℝ → ℝ) 
  (h_range : ∀ x, h x ∈ Set.Icc (-3) 5) 
  (k_range : ∀ x, k x ∈ Set.Icc (-1) 4) : 
  (∃ x y, h x * k y = 20) ∧ (∀ x y, h x * k y ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_functions_l3843_384335


namespace NUMINAMATH_CALUDE_sin_cos_2alpha_l3843_384343

def fixed_point : ℝ × ℝ := (4, 2)

def is_on_terminal_side (α : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = fixed_point.1 ∧ r * (Real.sin α) = fixed_point.2

theorem sin_cos_2alpha (α : ℝ) (h : is_on_terminal_side α) : 
  Real.sin (2 * α) + Real.cos (2 * α) = 7/5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_2alpha_l3843_384343


namespace NUMINAMATH_CALUDE_least_possible_difference_l3843_384357

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 3 → 
  ∀ w, (∃ a b c : ℤ, Even a ∧ Odd b ∧ Odd c ∧ a < b ∧ b < c ∧ b - a > 3 ∧ c - a = w) → w ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l3843_384357


namespace NUMINAMATH_CALUDE_evelyns_bottle_caps_l3843_384394

/-- The problem of Evelyn's bottle caps -/
theorem evelyns_bottle_caps
  (initial : ℕ)            -- Initial number of bottle caps
  (found : ℕ)              -- Number of bottle caps found
  (total : ℕ)              -- Total number of bottle caps at the end
  (h1 : found = 63)        -- Evelyn found 63 bottle caps
  (h2 : total = 81)        -- Evelyn ended up with 81 bottle caps in total
  (h3 : total = initial + found) -- The total is the sum of initial and found bottle caps
  : initial = 18 :=
by sorry

end NUMINAMATH_CALUDE_evelyns_bottle_caps_l3843_384394


namespace NUMINAMATH_CALUDE_points_two_units_from_negative_three_l3843_384362

theorem points_two_units_from_negative_three :
  ∀ x : ℝ, |(-3) - x| = 2 ↔ x = -5 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_points_two_units_from_negative_three_l3843_384362


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l3843_384354

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (fun (x y : ℝ) => x * Real.rpow 3 (1/4) + y * Real.rpow 5 (1/3)) ^ 400
  let total_terms := 401
  let rational_coeff_count := Finset.filter (fun k => 
    (k % 4 = 0) ∧ ((400 - k) % 3 = 0)
  ) (Finset.range (total_terms))
  34

#check rational_coefficient_terms_count

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l3843_384354


namespace NUMINAMATH_CALUDE_circle_equation_l3843_384396

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  x + 2*y = 0

theorem circle_equation :
  ∀ a : ℝ,
  a < 0 →
  (∃ x y : ℝ, circle_C a x y ∧ tangent_line x y) →
  (∀ x y : ℝ, circle_C a x y ↔ (x + 5)^2 + y^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3843_384396


namespace NUMINAMATH_CALUDE_percentage_relationship_l3843_384314

theorem percentage_relationship (a b : ℝ) (h : a = 1.5 * b) :
  3 * b = 2 * a := by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3843_384314


namespace NUMINAMATH_CALUDE_books_sold_l3843_384312

/-- Given the initial number of books and the remaining number of books,
    prove that the number of books sold is their difference. -/
theorem books_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) 
    (h1 : initial = 115)
    (h2 : remaining = 37)
    (h3 : sold = initial - remaining) : 
  sold = 78 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l3843_384312


namespace NUMINAMATH_CALUDE_closet_probability_l3843_384352

/-- The number of shirts in the closet -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the closet -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the closet -/
def num_socks : ℕ := 7

/-- The total number of articles of clothing in the closet -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles to be drawn -/
def draw_count : ℕ := 4

/-- The probability of drawing 2 shirts, 1 pair of shorts, and 1 pair of socks -/
theorem closet_probability : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) / 
  Nat.choose total_articles draw_count = 56 / 399 := by
  sorry

end NUMINAMATH_CALUDE_closet_probability_l3843_384352


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_10_l3843_384392

theorem last_digit_of_3_power_10 : ∃ n : ℕ, 3^10 ≡ 9 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_10_l3843_384392


namespace NUMINAMATH_CALUDE_incorrect_operation_l3843_384300

theorem incorrect_operation (a b : ℝ) : (-a^3)^2 * (-b^2)^3 = -a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l3843_384300


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3843_384345

/-- The line l with equation y = kx - 3k intersects the circle C with equation x^2 + y^2 - 4x = 0 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * x - 3 * k ∧ x^2 + y^2 - 4 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3843_384345


namespace NUMINAMATH_CALUDE_line_points_k_value_l3843_384378

/-- Given a line containing the points (0, 4), (7, k), and (21, -2), prove that k = 2 -/
theorem line_points_k_value (k : ℝ) : 
  (∀ t : ℝ, ∃ x y : ℝ, x = t * 7 ∧ y = t * (k - 4) + 4) → 
  (∃ t : ℝ, 21 = t * 7 ∧ -2 = t * (k - 4) + 4) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3843_384378


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3843_384358

/-- The function f(x) = a^(x-2015) + 2015 passes through the point (2015, 2016) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2015) + 2015
  f 2015 = 2016 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3843_384358


namespace NUMINAMATH_CALUDE_gcd_and_polynomial_evaluation_l3843_384332

theorem gcd_and_polynomial_evaluation :
  (Nat.gcd 72 168 = 24) ∧
  (Nat.gcd 98 280 = 14) ∧
  (let f : ℤ → ℤ := fun x => x^5 + x^3 + x^2 + x + 1;
   f 3 = 283) := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_polynomial_evaluation_l3843_384332


namespace NUMINAMATH_CALUDE_log15_12_equals_fraction_l3843_384349

-- Define the logarithm base 10 (lg) and logarithm base 15
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log15 (x : ℝ) := Real.log x / Real.log 15

-- State the theorem
theorem log15_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log15 12 = (2*a + b) / (1 - a + b) := by sorry

end NUMINAMATH_CALUDE_log15_12_equals_fraction_l3843_384349


namespace NUMINAMATH_CALUDE_charles_whistle_count_l3843_384313

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l3843_384313


namespace NUMINAMATH_CALUDE_line_equation_proof_l3843_384311

theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let line_eq := (y - point.2 = slope * (x - point.1))
  line_eq ↔ (y - 1 = Real.sqrt 3 * (x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3843_384311


namespace NUMINAMATH_CALUDE_train_speed_problem_l3843_384301

/-- Proves that given the conditions of the train problem, the speeds of the slower and faster trains are 60 km/hr and 70 km/hr respectively. -/
theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (remaining_distance : ℝ)
  (h1 : distance = 300)
  (h2 : time = 2)
  (h3 : speed_diff = 10)
  (h4 : remaining_distance = 40) :
  ∃ (v1 v2 : ℝ), v1 = 60 ∧ v2 = 70 ∧ v2 = v1 + speed_diff ∧
  distance - remaining_distance = (v1 + v2) * time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3843_384301


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_48_l3843_384393

/-- The number of ways to seat 7 people around a round table with constraints -/
def seating_arrangements : ℕ :=
  let total_people : ℕ := 7
  let fixed_people : ℕ := 3  -- Alice, Bob, and Carol
  let remaining_people : ℕ := total_people - fixed_people
  let ways_to_arrange_bob_and_carol : ℕ := 2
  ways_to_arrange_bob_and_carol * (Nat.factorial remaining_people)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 : seating_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_48_l3843_384393


namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l3843_384385

/-- Given a triangle ABC with internal angles A, B, and C, and vectors m and n
    satisfying certain conditions, prove that C = 2π/3 -/
theorem triangle_angle_c_value 
  (A B C : ℝ) 
  (triangle_sum : A + B + C = π)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (m_def : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (n_def : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (dot_product : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B)) :
  C = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l3843_384385


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l3843_384379

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  -(1 : ℚ) / n = -(1 / n) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l3843_384379


namespace NUMINAMATH_CALUDE_weight_of_b_l3843_384380

theorem weight_of_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 46) :
  B = 37 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l3843_384380


namespace NUMINAMATH_CALUDE_max_value_theorem_l3843_384388

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3843_384388


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3843_384330

-- Define sets A and B
def A : Set ℝ := {x | 1 / x ≥ 1}
def B : Set ℝ := {x | Real.log (1 - x) ≤ 0}

-- Theorem statement
theorem not_sufficient_not_necessary : 
  ¬(∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3843_384330


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l3843_384377

/-- Proves that the remainder yards after 15 marathons is 1500 --/
theorem marathon_remainder_yards :
  let marathons : ℕ := 15
  let miles_per_marathon : ℕ := 26
  let yards_per_marathon : ℕ := 385
  let yards_per_mile : ℕ := 1760
  let total_yards := marathons * (miles_per_marathon * yards_per_mile + yards_per_marathon)
  let full_miles := total_yards / yards_per_mile
  let remainder_yards := total_yards % yards_per_mile
  remainder_yards = 1500 := by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l3843_384377


namespace NUMINAMATH_CALUDE_symmetry_implies_a_eq_neg_one_l3843_384384

/-- A function f is symmetric about the line x = c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetry_implies_a_eq_neg_one :
  let f := fun (x : ℝ) => Real.sin (2 * x) + a * Real.cos (2 * x)
  SymmetricAbout f (-π/8) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_eq_neg_one_l3843_384384


namespace NUMINAMATH_CALUDE_trapezoid_area_l3843_384339

/-- A trapezoid with the given properties -/
structure Trapezoid where
  /-- Length of one diagonal -/
  diagonal1 : ℝ
  /-- Length of the other diagonal -/
  diagonal2 : ℝ
  /-- Length of the segment connecting the midpoints of the bases -/
  midpoint_segment : ℝ
  /-- The first diagonal is 3 -/
  h1 : diagonal1 = 3
  /-- The second diagonal is 5 -/
  h2 : diagonal2 = 5
  /-- The segment connecting the midpoints of the bases is 2 -/
  h3 : midpoint_segment = 2

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := 6

/-- Theorem stating that the area of the trapezoid with the given properties is 6 -/
theorem trapezoid_area (t : Trapezoid) : area t = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3843_384339


namespace NUMINAMATH_CALUDE_count_valid_quadruples_l3843_384346

def valid_quadruple (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
  a^2 + b^2 + c^2 + d^2 = 9 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81

theorem count_valid_quadruples :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ q ∈ s, valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2) ∧
    (∀ a b c d, valid_quadruple a b c d → (a, b, c, d) ∈ s) ∧
    s.card = 15 :=
sorry

end NUMINAMATH_CALUDE_count_valid_quadruples_l3843_384346


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3843_384364

theorem min_value_of_expression (x : ℝ) :
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧
  ∀ ε > 0, ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3843_384364


namespace NUMINAMATH_CALUDE_preimage_of_five_one_l3843_384318

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_five_one :
  ∃ (p : ℝ × ℝ), f p = (5, 1) ∧ p = (2, 3) :=
by
  sorry

end NUMINAMATH_CALUDE_preimage_of_five_one_l3843_384318


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3843_384386

/-- Given a quadratic function f(x) = ax^2 + bx where a > 0 and b > 0,
    if the slope of the tangent line at x = 1 is 2,
    then the minimum value of (8a + b) / (ab) is 9. -/
theorem quadratic_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a + b = 2) → (∀ x y : ℝ, x > 0 ∧ y > 0 → (8 * x + y) / (x * y) ≥ 9) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8 * x + y) / (x * y) = 9) := by
  sorry

#check quadratic_function_minimum

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3843_384386


namespace NUMINAMATH_CALUDE_garden_perimeter_l3843_384333

theorem garden_perimeter (garden_width playground_length playground_width : ℝ) : 
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
  2 * (garden_width + (playground_length * playground_width / garden_width)) = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3843_384333


namespace NUMINAMATH_CALUDE_f_properties_l3843_384307

noncomputable def f (x : ℝ) := x^2 / Real.log x

theorem f_properties :
  let e := Real.exp 1
  ∀ x ∈ Set.Icc (Real.exp (1/4)) e,
    (∀ y ∈ Set.Icc (Real.exp (1/4)) e, f y ≤ f e) ∧
    (f (Real.sqrt e) ≤ f x) ∧
    (∃ t ∈ Set.Icc (2/(e^2)) (1/e), 
      (∃ x₁ ∈ Set.Icc (1/e) 1, t * f x₁ = x₁) ∧
      (∃ x₂ ∈ Set.Ioc 1 (e^2), t * f x₂ = x₂) ∧
      (∀ s < 2/(e^2), ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂) ∧
      (∀ s ≥ 1/e, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3843_384307


namespace NUMINAMATH_CALUDE_equation_solution_l3843_384359

theorem equation_solution : ∃ y : ℝ, (16 : ℝ) ^ (2 * y - 4) = (1 / 4 : ℝ) ^ (5 - y) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3843_384359


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3843_384360

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - 2 * (a - 1) * x + a ≤ 0}
  (a > 1/2 → solution_set = ∅) ∧
  (a = 1/2 → solution_set = {-1}) ∧
  (0 < a ∧ a < 1/2 → solution_set = Set.Icc ((a - 1 - Real.sqrt (1 - 2*a)) / a) ((a - 1 + Real.sqrt (1 - 2*a)) / a)) ∧
  (a = 0 → solution_set = Set.Iic 0) ∧
  (a < 0 → solution_set = Set.Iic ((a - 1 + Real.sqrt (1 - 2*a)) / a) ∪ Set.Ici ((a - 1 - Real.sqrt (1 - 2*a)) / a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3843_384360


namespace NUMINAMATH_CALUDE_walters_pocket_percentage_l3843_384303

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Walter has -/
def num_pennies : ℕ := 2

/-- The number of nickels Walter has -/
def num_nickels : ℕ := 3

/-- The number of dimes Walter has -/
def num_dimes : ℕ := 2

/-- The total value of coins in Walter's pocket in cents -/
def total_value : ℕ := 
  num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

/-- The percentage of one dollar that Walter has in his pocket -/
theorem walters_pocket_percentage :
  (total_value : ℚ) / 100 * 100 = 37 := by sorry

end NUMINAMATH_CALUDE_walters_pocket_percentage_l3843_384303


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3843_384319

def expression : ℕ := 18^3 + 15^4 - 3^7

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3843_384319


namespace NUMINAMATH_CALUDE_system_solution_existence_l3843_384365

theorem system_solution_existence (b : ℝ) :
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*Real.sqrt 2 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3843_384365


namespace NUMINAMATH_CALUDE_group_c_marks_is_four_l3843_384374

/-- Represents the examination setup with three groups of questions -/
structure Examination where
  total_questions : ℕ
  group_a_marks : ℕ
  group_b_marks : ℕ
  group_b_questions : ℕ
  group_c_questions : ℕ

/-- Theorem stating that under the given conditions, each question in group C carries 4 marks -/
theorem group_c_marks_is_four (exam : Examination)
  (h_total : exam.total_questions = 100)
  (h_group_a : exam.group_a_marks = 1)
  (h_group_b : exam.group_b_marks = 2)
  (h_group_b_count : exam.group_b_questions = 23)
  (h_group_c_count : exam.group_c_questions = 1)
  (h_group_a_percentage : 
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             4 * exam.group_c_questions)) :
  ∃ (group_c_marks : ℕ), group_c_marks = 4 ∧
    group_c_marks > exam.group_b_marks ∧
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             group_c_marks * exam.group_c_questions) := by
  sorry

end NUMINAMATH_CALUDE_group_c_marks_is_four_l3843_384374


namespace NUMINAMATH_CALUDE_alternate_seating_four_boys_three_girls_l3843_384340

/-- The number of ways to seat 4 boys and 3 girls in a row alternately -/
def alternate_seating (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys = 4 ∧ num_girls = 3 then
    2 * (Nat.factorial num_boys * Nat.factorial num_girls)
  else
    0

theorem alternate_seating_four_boys_three_girls :
  alternate_seating 4 3 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_four_boys_three_girls_l3843_384340


namespace NUMINAMATH_CALUDE_intersection_M_N_l3843_384326

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3843_384326


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3843_384305

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3843_384305


namespace NUMINAMATH_CALUDE_martin_correct_is_40_l3843_384336

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of additional questions Kelsey answered correctly compared to Campbell -/
def kelsey_additional : ℕ := 8

/-- The number of fewer questions Martin answered correctly compared to Kelsey -/
def martin_fewer : ℕ := 3

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := campbell_correct + kelsey_additional - martin_fewer

theorem martin_correct_is_40 : martin_correct = 40 := by
  sorry

end NUMINAMATH_CALUDE_martin_correct_is_40_l3843_384336


namespace NUMINAMATH_CALUDE_officer_3_years_shoe_price_l3843_384390

def full_price : ℝ := 85
def discount_1_year : ℝ := 0.2
def discount_3_years : ℝ := 0.25

def price_after_1_year_discount : ℝ := full_price * (1 - discount_1_year)
def price_after_3_years_discount : ℝ := price_after_1_year_discount * (1 - discount_3_years)

theorem officer_3_years_shoe_price :
  price_after_3_years_discount = 51 :=
sorry

end NUMINAMATH_CALUDE_officer_3_years_shoe_price_l3843_384390


namespace NUMINAMATH_CALUDE_discount_clinic_visits_prove_discount_clinic_visits_l3843_384323

def normal_doctor_charge : ℝ := 200
def discount_percentage : ℝ := 0.7
def savings : ℝ := 80

theorem discount_clinic_visits : ℝ :=
  let discount_clinic_charge := normal_doctor_charge * (1 - discount_percentage)
  let total_paid := normal_doctor_charge - savings
  total_paid / discount_clinic_charge

theorem prove_discount_clinic_visits :
  discount_clinic_visits = 2 := by sorry

end NUMINAMATH_CALUDE_discount_clinic_visits_prove_discount_clinic_visits_l3843_384323


namespace NUMINAMATH_CALUDE_david_found_correct_l3843_384383

/-- The amount of money David found on the street -/
def david_found : ℕ := 12

/-- The initial amount of money Evan had -/
def evan_initial : ℕ := 1

/-- The cost of the watch -/
def watch_cost : ℕ := 20

/-- The amount Evan still needs after receiving money from David -/
def evan_still_needs : ℕ := 7

/-- Theorem stating that the amount David found is correct -/
theorem david_found_correct : 
  david_found = watch_cost - evan_still_needs - evan_initial :=
by sorry

end NUMINAMATH_CALUDE_david_found_correct_l3843_384383


namespace NUMINAMATH_CALUDE_marble_solution_l3843_384350

/-- Represents the number of marbles each person has -/
structure Marbles where
  selma : ℕ
  merill : ℕ
  elliot : ℕ
  vivian : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.selma = 50 ∧
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.vivian = m.merill + m.elliot + 10

/-- The theorem stating the solution to the marble problem -/
theorem marble_solution (m : Marbles) (h : marble_conditions m) : 
  m.merill = 30 ∧ m.vivian = 55 := by
  sorry

#check marble_solution

end NUMINAMATH_CALUDE_marble_solution_l3843_384350


namespace NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3843_384398

def football_gear_cost (x : ℝ) : Prop :=
  let shorts := x
  let tshirt := x
  let boots := 4 * x
  let shin_guards := 2 * x
  (shorts + tshirt = 2 * x) ∧
  (shorts + boots = 5 * x) ∧
  (shorts + shin_guards = 3 * x) ∧
  (shorts + tshirt + boots + shin_guards = 8 * x)

theorem total_cost_is_eight_times_shorts :
  ∀ x : ℝ, x > 0 → football_gear_cost x :=
by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3843_384398


namespace NUMINAMATH_CALUDE_cost_of_pens_l3843_384389

/-- Given that 150 pens cost $45, prove that 3300 pens cost $990 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℚ) (desired_amount : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_amount = 3300 →
  (desired_amount : ℚ) * (pack_cost / pack_size) = 990 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pens_l3843_384389


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l3843_384397

theorem subtraction_of_decimals :
  25.52 - 3.248 - 1.004 = 21.268 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l3843_384397


namespace NUMINAMATH_CALUDE_remainder_squared_pred_l3843_384324

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_pred_l3843_384324


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3843_384338

theorem equal_roots_quadratic (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) →
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3843_384338


namespace NUMINAMATH_CALUDE_max_factors_b_power_n_l3843_384387

def count_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_b_power_n (b n : ℕ+) (h1 : b ≤ 20) (h2 : n = 10) :
  (∃ (b' : ℕ+), b' ≤ 20 ∧ count_factors b' n = 231) ∧
  (∀ (b' : ℕ+), b' ≤ 20 → count_factors b' n ≤ 231) :=
sorry

end NUMINAMATH_CALUDE_max_factors_b_power_n_l3843_384387


namespace NUMINAMATH_CALUDE_min_delivery_time_l3843_384310

theorem min_delivery_time (n : Nat) (hn : n = 63) :
  let S := Fin n → Fin n
  (∃ (f : S), Function.Bijective f) →
  (∀ (f : S), Function.Bijective f →
    (∃ (i : Fin n), (i.val + 1) * (f i).val + 1 ≥ 1024)) ∧
  (∃ (f : S), Function.Bijective f ∧
    ∀ (i : Fin n), (i.val + 1) * (f i).val + 1 ≤ 1024) :=
by sorry

end NUMINAMATH_CALUDE_min_delivery_time_l3843_384310


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3843_384320

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 20*x - 8

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 44 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3843_384320


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3843_384399

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter + 4 }

/-- The theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial_config : TileConfiguration)
  (h1 : initial_config.tiles = 8)
  (h2 : initial_config.perimeter = 14) :
  ∃ (final_config : TileConfiguration),
    final_config = add_tiles initial_config 2 ∧
    final_config.perimeter = 18 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3843_384399


namespace NUMINAMATH_CALUDE_decrypt_ciphertext_l3843_384306

-- Define the encryption function
def encrypt (x : ℕ) : ℕ := 2^x - 2

-- State the theorem
theorem decrypt_ciphertext (y : ℕ) : 
  y = 1022 → ∃ x : ℕ, encrypt x = y ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_decrypt_ciphertext_l3843_384306


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l3843_384353

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
  c + i = 30 →
  3 * c - 2 * i = 55 →
  c = 23 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l3843_384353


namespace NUMINAMATH_CALUDE_tangent_roots_sine_cosine_ratio_l3843_384347

theorem tangent_roots_sine_cosine_ratio (α β p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (Real.sin (α + β)) / (Real.cos (α - β)) = -p / (q + 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_roots_sine_cosine_ratio_l3843_384347


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l3843_384363

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposites. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2 ∧ A.1 = -B.1

theorem symmetry_implies_sum_power (m n : ℝ) :
  symmetric_y_axis (m, 3) (4, n) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l3843_384363


namespace NUMINAMATH_CALUDE_bakery_puzzle_l3843_384331

/-- Represents the cost of items in a bakery -/
structure BakeryCosts where
  pastry : ℚ
  cupcake : ℚ
  bagel : ℚ

/-- Represents a purchase at the bakery -/
structure Purchase where
  pastries : ℕ
  cupcakes : ℕ
  bagels : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (costs : BakeryCosts) (purchase : Purchase) : ℚ :=
  costs.pastry * purchase.pastries + costs.cupcake * purchase.cupcakes + costs.bagel * purchase.bagels

theorem bakery_puzzle (costs : BakeryCosts) : 
  let petya := Purchase.mk 1 2 3
  let anya := Purchase.mk 3 0 1
  let kolya := Purchase.mk 0 6 0
  let lena := Purchase.mk 2 0 2
  totalCost costs petya = totalCost costs anya ∧ 
  totalCost costs anya = totalCost costs kolya → 
  totalCost costs lena = totalCost costs (Purchase.mk 0 5 0) := by
  sorry


end NUMINAMATH_CALUDE_bakery_puzzle_l3843_384331


namespace NUMINAMATH_CALUDE_anita_gave_five_apples_l3843_384370

/-- Represents the number of apples Tessa initially had -/
def initial_apples : ℕ := 4

/-- Represents the number of apples Tessa now has -/
def current_apples : ℕ := 9

/-- Represents the number of apples Anita gave Tessa -/
def apples_from_anita : ℕ := current_apples - initial_apples

theorem anita_gave_five_apples : apples_from_anita = 5 := by
  sorry

end NUMINAMATH_CALUDE_anita_gave_five_apples_l3843_384370


namespace NUMINAMATH_CALUDE_fraction_value_l3843_384322

theorem fraction_value (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3843_384322


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3843_384317

/-- Represents an ellipse with equation x²/16 + y²/9 = 1 -/
structure StandardEllipse where
  a : ℝ := 4
  b : ℝ := 3

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ

/-- Represents a focus of the ellipse -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The perimeter of triangle DEF₂ is 16 -/
theorem ellipse_triangle_perimeter
  (e : StandardEllipse)
  (F₁ F F₂ : Focus)
  (D E : EllipsePoint)
  (h1 : F₁.x < F.x) -- F₁ is the left focus
  (h2 : F₂ = F) -- F₂ is the right focus
  (h3 : D.x^2/16 + D.y^2/9 = 1) -- D is on the ellipse
  (h4 : E.x^2/16 + E.y^2/9 = 1) -- E is on the ellipse
  (h5 : ∃ (t : ℝ), D.x = (1-t)*F₁.x + t*E.x ∧ D.y = (1-t)*F₁.y + t*E.y) -- DE passes through F₁
  : (abs (D.x - F₁.x) + abs (D.y - F₁.y)) + 
    (abs (D.x - F₂.x) + abs (D.y - F₂.y)) +
    (abs (E.x - F₁.x) + abs (E.y - F₁.y)) + 
    (abs (E.x - F₂.x) + abs (E.y - F₂.y)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3843_384317


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3843_384356

theorem product_of_sum_and_cube_sum (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p^3 + q^3 = 370) : 
  p * q = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3843_384356


namespace NUMINAMATH_CALUDE_fred_dimes_l3843_384325

/-- Proves that if Fred has 90 cents and each dime is worth 10 cents, then Fred has 9 dimes -/
theorem fred_dimes (total_cents : ℕ) (dime_value : ℕ) (h1 : total_cents = 90) (h2 : dime_value = 10) :
  total_cents / dime_value = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_l3843_384325


namespace NUMINAMATH_CALUDE_distinct_roots_sum_l3843_384351

theorem distinct_roots_sum (a b c : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * (a - 6) = 7 →
  b * (b - 6) = 7 →
  c * (c - 6) = 7 →
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_l3843_384351


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3843_384373

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3843_384373


namespace NUMINAMATH_CALUDE_log_monotonic_l3843_384316

-- Define the logarithmic function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_monotonic : 
  ∀ x y : ℝ, x > 0 → y > 0 → x < y → log x < log y :=
by
  sorry

end NUMINAMATH_CALUDE_log_monotonic_l3843_384316


namespace NUMINAMATH_CALUDE_end_with_one_piece_l3843_384329

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (n : ℕ)
  (pieces : ℕ)

/-- Represents a valid move on the chessboard -/
inductive ValidMove : ChessboardState → ChessboardState → Prop
  | jump {s1 s2 : ChessboardState} :
      s1.n = s2.n ∧ s1.pieces = s2.pieces + 1 → ValidMove s1 s2

/-- Represents a sequence of valid moves -/
def ValidMoveSequence : ChessboardState → ChessboardState → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem stating the condition for ending with one piece -/
theorem end_with_one_piece (n : ℕ) :
  (∃ (final : ChessboardState),
    ValidMoveSequence (ChessboardState.mk n (n^2)) final ∧
    final.pieces = 1) ↔ n % 3 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_end_with_one_piece_l3843_384329


namespace NUMINAMATH_CALUDE_fisherman_multiple_is_three_l3843_384302

/-- The multiple of fish caught by the fisherman compared to the pelican and kingfisher combined -/
def fisherman_multiple (pelican_fish kingfisher_fish fisherman_fish : ℕ) : ℚ :=
  fisherman_fish / (pelican_fish + kingfisher_fish)

/-- Theorem stating the multiple of fish caught by the fisherman -/
theorem fisherman_multiple_is_three :
  ∀ (pelican_fish kingfisher_fish fisherman_fish : ℕ),
    pelican_fish = 13 →
    kingfisher_fish = pelican_fish + 7 →
    fisherman_fish = pelican_fish + 86 →
    fisherman_multiple pelican_fish kingfisher_fish fisherman_fish = 3 := by
  sorry

#eval fisherman_multiple 13 20 99

end NUMINAMATH_CALUDE_fisherman_multiple_is_three_l3843_384302
