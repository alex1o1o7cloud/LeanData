import Mathlib

namespace fraction_problem_l3294_329472

theorem fraction_problem (f : ℚ) : f * 76 = 76 - 19 → f = 3/4 := by
  sorry

end fraction_problem_l3294_329472


namespace correct_cases_delivered_l3294_329401

/-- The number of tins in each case -/
def tins_per_case : ℕ := 24

/-- The percentage of undamaged tins -/
def undamaged_percentage : ℚ := 95/100

/-- The number of undamaged tins left -/
def undamaged_tins : ℕ := 342

/-- The number of cases delivered -/
def cases_delivered : ℕ := 15

theorem correct_cases_delivered :
  cases_delivered * tins_per_case * undamaged_percentage = undamaged_tins := by
  sorry

end correct_cases_delivered_l3294_329401


namespace smallest_population_satisfying_conditions_l3294_329481

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem smallest_population_satisfying_conditions :
  ∃ (n : ℕ),
    (is_perfect_square n) ∧
    (is_perfect_square (n + 100)) ∧
    (∃ k : ℕ, n + 50 = k * k + 1) ∧
    (n % 3 = 0) ∧
    (∀ m : ℕ, m < n →
      ¬(is_perfect_square m ∧
        is_perfect_square (m + 100) ∧
        (∃ k : ℕ, m + 50 = k * k + 1) ∧
        (m % 3 = 0))) ∧
    n = 576 :=
by sorry

end smallest_population_satisfying_conditions_l3294_329481


namespace ratio_equality_l3294_329419

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end ratio_equality_l3294_329419


namespace repeating_decimal_problem_l3294_329408

/-- Represents a repeating decimal with a single digit followed by 25 -/
def RepeatingDecimal (d : Nat) : ℚ :=
  (d * 100 + 25 : ℚ) / 999

/-- The main theorem -/
theorem repeating_decimal_problem (n : ℕ) (d : Nat) 
    (h_n_pos : n > 0)
    (h_d_digit : d < 10)
    (h_eq : (n : ℚ) / 810 = RepeatingDecimal d) :
    n = 750 ∧ d = 9 := by
  sorry

end repeating_decimal_problem_l3294_329408


namespace son_age_l3294_329477

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 37 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end son_age_l3294_329477


namespace parallel_vectors_x_value_l3294_329431

/-- Two planar vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given two parallel planar vectors (3, 1) and (x, -3), x equals -9 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (3, 1) (x, -3) → x = -9 := by
  sorry

end parallel_vectors_x_value_l3294_329431


namespace tangent_curve_sum_l3294_329486

/-- The curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1).
    This theorem proves that b + c = -2. -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2*x^2 + b*x + c = x - 3 → x = 2) →  -- Tangent condition
  -2*2^2 + b*2 + c = -1 →                    -- Point (2, -1) lies on the curve
  2 - 3 = -1 →                               -- Point (2, -1) lies on the line
  (-4*2 + b = 1) →                           -- Derivative equality at x = 2
  b + c = -2 := by
sorry

end tangent_curve_sum_l3294_329486


namespace square_impossibility_l3294_329466

theorem square_impossibility (n : ℕ) : n^2 = 24 → False := by
  sorry

end square_impossibility_l3294_329466


namespace even_sum_probability_l3294_329475

/-- Represents a wheel with sections --/
structure Wheel where
  totalSections : Nat
  evenSections : Nat
  oddSections : Nat
  zeroSections : Nat

/-- First wheel configuration --/
def wheel1 : Wheel := {
  totalSections := 6
  evenSections := 2
  oddSections := 3
  zeroSections := 1
}

/-- Second wheel configuration --/
def wheel2 : Wheel := {
  totalSections := 4
  evenSections := 2
  oddSections := 2
  zeroSections := 0
}

/-- Calculate the probability of getting an even sum when spinning two wheels --/
def probabilityEvenSum (w1 w2 : Wheel) : Real :=
  sorry

/-- Theorem: The probability of getting an even sum when spinning the two given wheels is 1/2 --/
theorem even_sum_probability :
  probabilityEvenSum wheel1 wheel2 = 1/2 := by
  sorry

end even_sum_probability_l3294_329475


namespace club_members_neither_subject_l3294_329489

theorem club_members_neither_subject (total : ℕ) (cs : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 80)
  (h3 : bio = 50)
  (h4 : both = 15) :
  total - (cs + bio - both) = 35 := by
  sorry

end club_members_neither_subject_l3294_329489


namespace perimeter_plus_area_sum_l3294_329443

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ
  is_parallelogram : v1.1 + v3.1 = v2.1 + v4.1 ∧ v1.2 + v3.2 = v2.2 + v4.2

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  2 * (dist p.v1 p.v2 + dist p.v2 p.v3)

/-- Calculate the area of a parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  abs ((p.v2.1 - p.v1.1) * (p.v3.2 - p.v1.2) - (p.v3.1 - p.v1.1) * (p.v2.2 - p.v1.2))

/-- The sum of perimeter and area for the specific parallelogram -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) 
  (h1 : p.v1 = (2, 3)) 
  (h2 : p.v2 = (5, 7)) 
  (h3 : p.v3 = (0, -1)) : 
  perimeter p + area p = 10 + 12 * Real.sqrt 5 := by
  sorry


end perimeter_plus_area_sum_l3294_329443


namespace gorilla_exhibit_percentage_is_80_l3294_329432

-- Define the given parameters
def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_visitors : ℕ := 320

-- Define the total number of visitors
def total_visitors : ℕ := visitors_per_hour * open_hours

-- Define the percentage of visitors going to the gorilla exhibit
def gorilla_exhibit_percentage : ℚ := (gorilla_exhibit_visitors : ℚ) / (total_visitors : ℚ) * 100

-- Theorem statement
theorem gorilla_exhibit_percentage_is_80 : 
  gorilla_exhibit_percentage = 80 := by sorry

end gorilla_exhibit_percentage_is_80_l3294_329432


namespace decimal_to_fraction_l3294_329433

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l3294_329433


namespace bathtub_fill_time_with_open_drain_l3294_329440

/-- Represents the time it takes to fill a bathtub with the drain open. -/
def fill_time_with_open_drain (fill_time drain_time : ℚ) : ℚ :=
  (fill_time * drain_time) / (drain_time - fill_time)

/-- Theorem stating that a bathtub taking 10 minutes to fill and 12 minutes to drain
    will take 60 minutes to fill with the drain open. -/
theorem bathtub_fill_time_with_open_drain :
  fill_time_with_open_drain 10 12 = 60 := by
  sorry

#eval fill_time_with_open_drain 10 12

end bathtub_fill_time_with_open_drain_l3294_329440


namespace intersection_product_equality_l3294_329450

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary operations and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (on_arc : Point → Point → Point → Circle → Prop)
variable (meets_at : Point → Point → Circle → Point → Prop)
variable (intersect_at : Point → Point → Point → Point → Point → Prop)
variable (length : Point → Point → ℝ)

-- Define the given points and circles
variable (O₁ O₂ : Circle)
variable (A B R T C D Q P E F : Point)

-- State the theorem
theorem intersection_product_equality
  (h1 : intersect O₁ O₂ A B)
  (h2 : on_arc A B R O₁)
  (h3 : on_arc A B T O₂)
  (h4 : meets_at A R O₂ C)
  (h5 : meets_at B R O₂ D)
  (h6 : meets_at A T O₁ Q)
  (h7 : meets_at B T O₁ P)
  (h8 : intersect_at P R T D E)
  (h9 : intersect_at Q R T C F) :
  length A E * length B T * length B R = length B F * length A T * length A R :=
sorry

end intersection_product_equality_l3294_329450


namespace age_ratio_l3294_329412

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.betty = 5

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) 
  (h : satisfiesConditions ages) : 
  ages.albert / ages.mary = 2 := by
sorry


end age_ratio_l3294_329412


namespace minimum_nickels_needed_l3294_329465

def book_cost : ℚ := 46.25

def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 5
def quarters : ℕ := 10

def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ,
    (n : ℚ) * nickel_value +
    (five_dollar_bills : ℚ) * 5 +
    (one_dollar_bills : ℚ) * 1 +
    (quarters : ℚ) * 0.25 ≥ book_cost ∧
    ∀ m : ℕ, m < n →
      (m : ℚ) * nickel_value +
      (five_dollar_bills : ℚ) * 5 +
      (one_dollar_bills : ℚ) * 1 +
      (quarters : ℚ) * 0.25 < book_cost :=
by
  sorry

end minimum_nickels_needed_l3294_329465


namespace total_population_l3294_329492

def wildlife_park (num_lions : ℕ) (num_leopards : ℕ) (num_adult_elephants : ℕ) (num_zebras : ℕ) : Prop :=
  (num_lions = 2 * num_leopards) ∧
  (num_adult_elephants = (num_lions * 3 / 4 + num_leopards * 3 / 5) / 2) ∧
  (num_zebras = num_adult_elephants + num_leopards) ∧
  (num_lions = 200)

theorem total_population (num_lions num_leopards num_adult_elephants num_zebras : ℕ) :
  wildlife_park num_lions num_leopards num_adult_elephants num_zebras →
  num_lions + num_leopards + (num_adult_elephants + 100) + num_zebras = 710 :=
by
  sorry

#check total_population

end total_population_l3294_329492


namespace worker_speed_comparison_l3294_329456

/-- Given two workers A and B, this theorem proves that A is 3 times faster than B
    under the specified conditions. -/
theorem worker_speed_comparison 
  (work_rate_A : ℝ) 
  (work_rate_B : ℝ) 
  (total_work : ℝ) 
  (h1 : work_rate_A + work_rate_B = total_work / 24)
  (h2 : work_rate_A = total_work / 32) :
  work_rate_A = 3 * work_rate_B :=
sorry

end worker_speed_comparison_l3294_329456


namespace max_value_constraint_l3294_329467

theorem max_value_constraint (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ x' y' : ℝ, 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end max_value_constraint_l3294_329467


namespace quadratic_roots_product_l3294_329434

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 14 := by
sorry

end quadratic_roots_product_l3294_329434


namespace books_sold_l3294_329461

theorem books_sold (initial_books remaining_books : ℕ) 
  (h1 : initial_books = 136) 
  (h2 : remaining_books = 27) : 
  initial_books - remaining_books = 109 := by
  sorry

end books_sold_l3294_329461


namespace intersection_M_N_l3294_329425

def M : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l3294_329425


namespace kyle_origami_stars_l3294_329491

theorem kyle_origami_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) :
  initial_bottles = 4 →
  additional_bottles = 5 →
  stars_per_bottle = 25 →
  (initial_bottles + additional_bottles) * stars_per_bottle = 225 := by
  sorry

end kyle_origami_stars_l3294_329491


namespace sum_reciprocals_equal_two_point_five_l3294_329490

theorem sum_reciprocals_equal_two_point_five
  (a b c d e : ℝ)
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1) (he : e ≠ -1)
  (ω : ℂ)
  (hω1 : ω^3 = 1)
  (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) + (1 / (e + ω)) = 5 / (2*ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) + (1 / (e + 1)) = 2.5 := by
  sorry

end sum_reciprocals_equal_two_point_five_l3294_329490


namespace juanico_age_30_years_from_now_l3294_329447

/-- Juanico's age 30 years from now, given the conditions in the problem -/
def juanico_future_age (gladys_current_age : ℕ) (juanico_current_age : ℕ) : ℕ :=
  juanico_current_age + 30

/-- The theorem stating Juanico's age 30 years from now -/
theorem juanico_age_30_years_from_now :
  ∀ (gladys_current_age : ℕ) (juanico_current_age : ℕ),
    gladys_current_age + 10 = 40 →
    juanico_current_age = gladys_current_age / 2 - 4 →
    juanico_future_age gladys_current_age juanico_current_age = 41 :=
by
  sorry

#check juanico_age_30_years_from_now

end juanico_age_30_years_from_now_l3294_329447


namespace sqrt_x_minus_one_real_l3294_329407

theorem sqrt_x_minus_one_real (x : ℝ) : x ≥ 1 ↔ ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end sqrt_x_minus_one_real_l3294_329407


namespace soap_brand_usage_l3294_329413

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_e : ℕ) (both : ℕ) :
  total = 200 →
  neither = 80 →
  only_e = 60 →
  total = neither + only_e + both + 3 * both →
  both = 15 := by
sorry

end soap_brand_usage_l3294_329413


namespace negation_of_universal_proposition_l3294_329452

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end negation_of_universal_proposition_l3294_329452


namespace product_difference_bound_l3294_329497

theorem product_difference_bound (x y a b m ε : ℝ) 
  (ε_pos : ε > 0) (m_pos : m > 0) 
  (h1 : |x - a| < ε / (2 * m))
  (h2 : |y - b| < ε / (2 * |a|))
  (h3 : 0 < y) (h4 : y < m) : 
  |x * y - a * b| < ε := by
sorry

end product_difference_bound_l3294_329497


namespace inverse_proportion_inequality_l3294_329439

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end inverse_proportion_inequality_l3294_329439


namespace bearings_count_proof_l3294_329414

/-- The number of machines -/
def num_machines : ℕ := 10

/-- The normal cost per ball bearing in cents -/
def normal_cost : ℕ := 100

/-- The sale price per ball bearing in cents -/
def sale_price : ℕ := 75

/-- The additional discount rate for bulk purchase -/
def bulk_discount : ℚ := 1/5

/-- The amount saved in cents by buying during the sale -/
def amount_saved : ℕ := 12000

/-- The number of ball bearings per machine -/
def bearings_per_machine : ℕ := 30

theorem bearings_count_proof :
  ∃ (x : ℕ),
    x = bearings_per_machine ∧
    (num_machines * normal_cost * x) -
    (num_machines * sale_price * x * (1 - bulk_discount)) =
    amount_saved :=
sorry

end bearings_count_proof_l3294_329414


namespace theater_ticket_sales_l3294_329430

/-- Theorem: Theater Ticket Sales --/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_revenue : ℕ)
  (balcony_excess : ℕ)
  (h1 : orchestra_price = 12)
  (h2 : balcony_price = 8)
  (h3 : total_revenue = 3320)
  (h4 : balcony_excess = 190)
  : ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = total_revenue ∧
    balcony_tickets = orchestra_tickets + balcony_excess ∧
    orchestra_tickets + balcony_tickets = 370 :=
by
  sorry


end theater_ticket_sales_l3294_329430


namespace range_of_a_l3294_329470

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x > -1, x^2 / (x + 1) ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 - a * x + 1 = 0

-- State the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a) ∧ ¬(q a)) ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
sorry

end range_of_a_l3294_329470


namespace nancy_total_games_l3294_329454

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating that Nancy will attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 8 7 = 24 := by
  sorry

end nancy_total_games_l3294_329454


namespace count_valid_sums_l3294_329462

/-- The number of valid ways to sum to 5750 using 5's, 55's, and 555's -/
def num_valid_sums : ℕ := 124

/-- Predicate for a valid sum configuration -/
def is_valid_sum (a b c : ℕ) : Prop :=
  a + 11 * b + 111 * c = 1150

/-- The length of the original string of 5's -/
def string_length (a b c : ℕ) : ℕ :=
  a + 2 * b + 3 * c

/-- Theorem stating that there are exactly 124 valid string lengths -/
theorem count_valid_sums :
  (∃ (S : Finset ℕ), S.card = num_valid_sums ∧
    (∀ n, n ∈ S ↔ ∃ a b c, is_valid_sum a b c ∧ string_length a b c = n)) :=
sorry

end count_valid_sums_l3294_329462


namespace expand_product_l3294_329485

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 10) = 2*x^2 + 23*x + 30 := by
  sorry

end expand_product_l3294_329485


namespace jose_alisson_difference_l3294_329493

/-- Represents the scores of three students in a test -/
structure TestScores where
  jose : ℕ
  meghan : ℕ
  alisson : ℕ

/-- Properties of the test and scores -/
def valid_scores (s : TestScores) : Prop :=
  s.meghan = s.jose - 20 ∧
  s.jose > s.alisson ∧
  s.jose = 90 ∧
  s.jose + s.meghan + s.alisson = 210

/-- Theorem stating the difference between Jose's and Alisson's scores -/
theorem jose_alisson_difference (s : TestScores) 
  (h : valid_scores s) : s.jose - s.alisson = 40 := by
  sorry

end jose_alisson_difference_l3294_329493


namespace perpendicular_vectors_l3294_329438

def a : Fin 2 → ℝ := ![3, 2]
def b (n : ℝ) : Fin 2 → ℝ := ![2, n]

theorem perpendicular_vectors (n : ℝ) : 
  (∀ i : Fin 2, (a i) * (b n i) = 0) → n = 3 := by
  sorry

end perpendicular_vectors_l3294_329438


namespace double_counted_page_l3294_329487

/-- Given a book with 62 pages, prove that if the sum of all page numbers
    plus an additional count of one page number equals 1997,
    then the page number that was counted twice is 44. -/
theorem double_counted_page (n : ℕ) (x : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + x = 1997 → 
  x = 44 := by
sorry

end double_counted_page_l3294_329487


namespace integer_equation_solution_l3294_329448

theorem integer_equation_solution (x y : ℤ) (h : x^2 + 2 = 3*x + 75*y) :
  ∃ t : ℤ, x = 75*t + 1 ∨ x = 75*t + 2 ∨ x = 75*t + 26 ∨ x = 75*t - 23 :=
sorry

end integer_equation_solution_l3294_329448


namespace inequality_solution_set_l3294_329495

theorem inequality_solution_set :
  {x : ℝ | (|x| + x) * (Real.sin x - 2) < 0} = Set.Ioo 0 Real.pi := by
  sorry

end inequality_solution_set_l3294_329495


namespace inscribed_rectangle_area_l3294_329424

/-- Triangle EFG with inscribed rectangle ABCD -/
structure InscribedRectangle where
  /-- Length of side EG of triangle EFG -/
  eg : ℝ
  /-- Height of altitude from F to EG -/
  altitude : ℝ
  /-- Length of side AD of rectangle ABCD -/
  ad : ℝ
  /-- Length of side AB of rectangle ABCD -/
  ab : ℝ
  /-- AD is on EG -/
  ad_on_eg : ad ≤ eg
  /-- AB is one-third of AD -/
  ab_is_third_of_ad : ab = ad / 3
  /-- EG is 15 inches -/
  eg_length : eg = 15
  /-- Altitude is 10 inches -/
  altitude_length : altitude = 10

/-- The area of the inscribed rectangle ABCD is 100/3 square inches -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) : 
  rect.ad * rect.ab = 100 / 3 := by
  sorry

end inscribed_rectangle_area_l3294_329424


namespace unique_right_triangle_l3294_329494

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triangle is right-angled -/
def Triangle.isRight (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2 ∨ t.b ^ 2 + t.c ^ 2 = t.a ^ 2 ∨ t.c ^ 2 + t.a ^ 2 = t.b ^ 2

/-- The main theorem -/
theorem unique_right_triangle :
  ∃! k : ℕ+, 
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧ 
    (∃ t : Triangle, t.a + t.b + t.c = 30 ∧ t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k ∧ t.isRight) :=
by sorry

end unique_right_triangle_l3294_329494


namespace max_cut_length_30x30_225_l3294_329426

/-- Represents a square board with side length and number of parts it's cut into -/
structure Board where
  side_length : ℕ
  num_parts : ℕ

/-- Calculates the maximum total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 parts -/
theorem max_cut_length_30x30_225 :
  let b : Board := { side_length := 30, num_parts := 225 }
  max_cut_length b = 1065 :=
sorry

end max_cut_length_30x30_225_l3294_329426


namespace not_in_sample_l3294_329445

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_total : total_students = 60
  h_sample : sample_size = 5
  h_known : known_seats = {3, 15, 45, 53}

/-- The interval between sampled items in systematic sampling -/
def sample_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given seat number could be in the sample -/
def could_be_in_sample (s : SystematicSampling) (seat : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ s.sample_size ∧ seat = k * (sample_interval s)

/-- The main theorem stating that 37 cannot be the remaining seat in the sample -/
theorem not_in_sample (s : SystematicSampling) : ¬(could_be_in_sample s 37) := by
  sorry

end not_in_sample_l3294_329445


namespace euclidean_division_37_by_5_l3294_329482

theorem euclidean_division_37_by_5 :
  ∃ (q r : ℤ), 37 = 5 * q + r ∧ 0 ≤ r ∧ r < 5 ∧ q = 7 ∧ r = 2 :=
by sorry

end euclidean_division_37_by_5_l3294_329482


namespace percentage_of_1000_l3294_329437

theorem percentage_of_1000 (x : ℝ) (h : x = 66.2) : 
  (x / 1000) * 100 = 6.62 := by
  sorry

end percentage_of_1000_l3294_329437


namespace min_value_of_a_l3294_329422

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
by sorry

end min_value_of_a_l3294_329422


namespace octopus_ink_conversion_l3294_329418

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (units : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + units * 8^0

/-- The octopus ink problem -/
theorem octopus_ink_conversion :
  base8ToBase10 2 7 6 = 190 := by
  sorry

end octopus_ink_conversion_l3294_329418


namespace pencil_distribution_l3294_329429

theorem pencil_distribution (total_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) :
  total_pencils = 24 →
  num_friends = 3 →
  total_pencils = num_friends * pencils_per_friend →
  pencils_per_friend = 8 := by
sorry

end pencil_distribution_l3294_329429


namespace rectangle_perimeter_equals_20_l3294_329403

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- Area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := 
  sorry

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.w * r.l

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.w + r.l)

theorem rectangle_perimeter_equals_20 (t : Triangle) (r : Rectangle) :
  t.a = 6 ∧ t.b = 8 ∧ t.c = 10 ∧ r.w = 4 ∧ Triangle.area t = Rectangle.area r →
  Rectangle.perimeter r = 20 := by
  sorry

end rectangle_perimeter_equals_20_l3294_329403


namespace genevieve_thermoses_l3294_329406

/-- Proves the number of thermoses Genevieve drank given the conditions -/
theorem genevieve_thermoses (total_coffee : ℚ) (num_thermoses : ℕ) (genevieve_consumption : ℚ) : 
  total_coffee = 4.5 ∧ num_thermoses = 18 ∧ genevieve_consumption = 6 →
  (genevieve_consumption / (total_coffee * 8 / num_thermoses) : ℚ) = 3 := by
  sorry

end genevieve_thermoses_l3294_329406


namespace parallel_line_plane_not_imply_parallel_lines_l3294_329453

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLine : Line → Line → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Define the specific objects
variable (a b : Line)
variable (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines
  (h1 : parallelLineToPlane a α)
  (h2 : lineInPlane b α) :
  ¬ (parallelLine a b) :=
sorry

end parallel_line_plane_not_imply_parallel_lines_l3294_329453


namespace hostel_stay_duration_l3294_329435

/-- Cost structure for a student youth hostel stay -/
structure CostStructure where
  first_week_rate : ℝ
  additional_day_rate : ℝ

/-- Calculate the number of days stayed given the cost structure and total cost -/
def days_stayed (cs : CostStructure) (total_cost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for the given cost structure and total cost, the stay is 23 days -/
theorem hostel_stay_duration :
  let cs : CostStructure := { first_week_rate := 18, additional_day_rate := 14 }
  let total_cost : ℝ := 350
  days_stayed cs total_cost = 23 := by
  sorry

end hostel_stay_duration_l3294_329435


namespace vector_linear_combination_l3294_329442

/-- Given vectors a, b, and c in R², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (-1, 2)) :
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end vector_linear_combination_l3294_329442


namespace digits_of_power_product_l3294_329457

theorem digits_of_power_product : 
  (Nat.log 10 (2^15 * 5^10) + 1 : ℕ) = 12 := by sorry

end digits_of_power_product_l3294_329457


namespace inequality_proof_l3294_329404

theorem inequality_proof (α : ℝ) (hα : α > 0) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y :=
by sorry

end inequality_proof_l3294_329404


namespace square_side_length_l3294_329417

theorem square_side_length (side : ℝ) : 
  (5 * side) * (side / 2) = 160 → side = 8 := by
  sorry

end square_side_length_l3294_329417


namespace sector_arc_length_l3294_329473

/-- The length of an arc in a circular sector -/
def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ := radius * centralAngle

theorem sector_arc_length :
  let radius : ℝ := 16
  let centralAngle : ℝ := 2
  arcLength radius centralAngle = 32 := by sorry

end sector_arc_length_l3294_329473


namespace tan_105_degrees_l3294_329463

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l3294_329463


namespace books_left_to_read_l3294_329400

theorem books_left_to_read (total_books assigned_books : ℕ) 
  (mcgregor_finished floyd_finished : ℕ) 
  (h1 : assigned_books = 89)
  (h2 : mcgregor_finished = 34)
  (h3 : floyd_finished = 32)
  (h4 : total_books = assigned_books - (mcgregor_finished + floyd_finished)) :
  total_books = 23 := by
  sorry

end books_left_to_read_l3294_329400


namespace problem_statements_l3294_329488

theorem problem_statements (a b : ℝ) :
  (ab < 0 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → a / b = -1) ∧
  (a + b < 0 ∧ ab > 0 → |2*a + 3*b| = -(2*a + 3*b)) ∧
  ¬(∀ a b : ℝ, |a - b| + a - b = 0 → b > a) ∧
  ¬(∀ a b : ℝ, |a| > |b| → (a + b) * (a - b) < 0) :=
by sorry

end problem_statements_l3294_329488


namespace number_puzzle_l3294_329436

theorem number_puzzle (x y : ℝ) : x = 95 → (x / 5 + y = 42) → y = 23 := by
  sorry

end number_puzzle_l3294_329436


namespace deposit_percentage_l3294_329498

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 140 →
  remaining = 1260 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end deposit_percentage_l3294_329498


namespace cube_surface_area_from_diagonal_l3294_329409

/-- The surface area of a cube with space diagonal length 6 is 72 -/
theorem cube_surface_area_from_diagonal (d : ℝ) (h : d = 6) : 
  6 * (d / Real.sqrt 3) ^ 2 = 72 := by
  sorry

end cube_surface_area_from_diagonal_l3294_329409


namespace transcendental_equation_solution_l3294_329471

-- Define the variables
variable (n : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- State the theorem
theorem transcendental_equation_solution (hx : x = 3) (hy : y = 27) 
  (h : Real.exp (n / (2 * Real.sqrt (Real.pi + x))) = y) :
  n ^ (n / (2 * Real.sqrt (Real.pi + 3))) = Real.exp 27 := by
  sorry


end transcendental_equation_solution_l3294_329471


namespace polygon_is_decagon_iff_seven_diagonals_l3294_329423

/-- A polygon is a decagon if and only if 7 diagonals can be drawn from a single vertex. -/
theorem polygon_is_decagon_iff_seven_diagonals (n : ℕ) : 
  n = 10 ↔ n - 3 = 7 :=
sorry

end polygon_is_decagon_iff_seven_diagonals_l3294_329423


namespace asterisk_equation_solution_l3294_329484

theorem asterisk_equation_solution :
  ∃! x : ℝ, x > 0 ∧ (x / 20) * (x / 180) = 1 :=
by
  -- The proof goes here
  sorry

end asterisk_equation_solution_l3294_329484


namespace range_of_a_l3294_329402

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5 * x - 6}
def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) :
  (B a ∩ C a = C a) ↔ (2 ≤ a ∧ a ≤ 3) := by sorry

end range_of_a_l3294_329402


namespace f_composition_eq_one_fourth_l3294_329441

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end f_composition_eq_one_fourth_l3294_329441


namespace cost_per_minute_is_twelve_cents_l3294_329460

/-- Calculates the cost per minute for a phone service -/
def costPerMinute (monthlyFee : ℚ) (totalBill : ℚ) (minutesUsed : ℕ) : ℚ :=
  (totalBill - monthlyFee) / minutesUsed

/-- Proof that the cost per minute is $0.12 given the specified conditions -/
theorem cost_per_minute_is_twelve_cents :
  let monthlyFee : ℚ := 2
  let totalBill : ℚ := 23.36
  let minutesUsed : ℕ := 178
  costPerMinute monthlyFee totalBill minutesUsed = 0.12 := by
  sorry

#eval costPerMinute 2 23.36 178

end cost_per_minute_is_twelve_cents_l3294_329460


namespace train_crossing_time_l3294_329496

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = platform_length →
  train_length = 750 →
  train_speed = 90 * 1000 / 3600 →
  (train_length + platform_length) / train_speed = 60 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3294_329496


namespace no_positive_integer_solutions_l3294_329458

theorem no_positive_integer_solutions :
  ¬∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end no_positive_integer_solutions_l3294_329458


namespace power_of_five_mod_ten_thousand_l3294_329405

theorem power_of_five_mod_ten_thousand : 5^2023 % 10000 = 3125 := by
  sorry

end power_of_five_mod_ten_thousand_l3294_329405


namespace prob_both_selected_l3294_329411

theorem prob_both_selected (prob_x prob_y prob_both : ℚ) : 
  prob_x = 1/5 → prob_y = 2/7 → prob_both = prob_x * prob_y → prob_both = 2/35 := by
  sorry

end prob_both_selected_l3294_329411


namespace no_integer_distance_point_l3294_329427

theorem no_integer_distance_point (x y : ℕ) (hx : Odd x) (hy : Odd y) :
  ¬ ∃ (a d : ℝ), 0 < a ∧ a < x ∧ 0 < d ∧ d < y ∧
    (∃ (w x y z : ℕ), 
      a^2 + d^2 = (w : ℝ)^2 ∧
      (x - a)^2 + d^2 = (x : ℝ)^2 ∧
      a^2 + (y - d)^2 = (y : ℝ)^2 ∧
      (x - a)^2 + (y - d)^2 = (z : ℝ)^2) :=
by sorry

end no_integer_distance_point_l3294_329427


namespace safe_gold_rows_l3294_329469

/-- The number of gold bars per row in the safe. -/
def gold_bars_per_row : ℕ := 20

/-- The total worth of all gold bars in the safe, in dollars. -/
def total_worth : ℕ := 1600000

/-- The number of rows of gold bars in the safe. -/
def num_rows : ℕ := total_worth / (gold_bars_per_row * (total_worth / gold_bars_per_row))

theorem safe_gold_rows : num_rows = 1 := by
  sorry

end safe_gold_rows_l3294_329469


namespace base_8_23456_equals_10030_l3294_329446

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_23456_equals_10030 :
  base_8_to_10 [2, 3, 4, 5, 6] = 10030 := by
  sorry

end base_8_23456_equals_10030_l3294_329446


namespace chess_and_go_purchase_l3294_329480

theorem chess_and_go_purchase (m : ℕ) : 
  (m + (120 - m) = 120) →
  (m ≥ 2 * (120 - m)) →
  (30 * m + 25 * (120 - m) ≤ 3500) →
  (80 ≤ m ∧ m ≤ 100) :=
by sorry

end chess_and_go_purchase_l3294_329480


namespace tangent_identity_l3294_329459

theorem tangent_identity (β : ℝ) : 
  Real.tan (6 * β) - Real.tan (4 * β) - Real.tan (2 * β) = 
  Real.tan (6 * β) * Real.tan (4 * β) * Real.tan (2 * β) := by
  sorry

end tangent_identity_l3294_329459


namespace train_crossing_time_l3294_329420

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  train_speed_kmh = 216 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
  sorry

end train_crossing_time_l3294_329420


namespace point_symmetric_second_quadrant_l3294_329455

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is symmetric about the y-axis -/
def isSymmetricAboutYAxis (p : Point) : Prop :=
  p.x < 0

/-- The main theorem -/
theorem point_symmetric_second_quadrant (a : ℝ) :
  let A : Point := ⟨a - 1, 2 * a - 4⟩
  isInSecondQuadrant A ∧ isSymmetricAboutYAxis A → a > 2 :=
by sorry

end point_symmetric_second_quadrant_l3294_329455


namespace rectangular_solid_surface_area_l3294_329444

/-- The total surface area of a rectangular solid. -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, 
    width 8 meters, and depth 5 meters is 314 square meters. -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end rectangular_solid_surface_area_l3294_329444


namespace complex_equation_solution_l3294_329479

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 2 * Complex.I * z = 1 + 5 * Complex.I * z ∧ z = -((4 : ℂ) * Complex.I / 7) := by
  sorry

end complex_equation_solution_l3294_329479


namespace function_properties_l3294_329416

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ sorry

-- State the main theorem
theorem function_properties (h : ∀ x : ℝ, 3 * f (2 - x) - 2 * f x = x^2 - 2*x) :
  (∀ x : ℝ, f x = x^2 - 2*x) ∧
  (∀ a : ℝ, a > 1 → ∀ x : ℝ, f x + a > 0) ∧
  (∀ x : ℝ, f x + 1 > 0 ↔ x ≠ 1) ∧
  (∀ a : ℝ, a < 1 → ∀ x : ℝ, f x + a > 0 ↔ x > 1 + Real.sqrt (1 - a) ∨ x < 1 - Real.sqrt (1 - a)) :=
by sorry

end function_properties_l3294_329416


namespace student_d_not_top_student_l3294_329476

/-- Represents a student's rankings in three consecutive exams -/
structure StudentRankings :=
  (r1 r2 r3 : ℕ)

/-- Calculates the mode of three numbers -/
def mode (a b c : ℕ) : ℕ := sorry

/-- Calculates the variance of three numbers -/
def variance (a b c : ℕ) : ℚ := sorry

/-- Determines if a student is a top student based on their rankings -/
def is_top_student (s : StudentRankings) : Prop :=
  s.r1 ≤ 3 ∧ s.r2 ≤ 3 ∧ s.r3 ≤ 3

theorem student_d_not_top_student (s : StudentRankings) :
  mode s.r1 s.r2 s.r3 = 2 ∧ variance s.r1 s.r2 s.r3 > 1 →
  ¬(is_top_student s) := by sorry

end student_d_not_top_student_l3294_329476


namespace power_function_not_in_fourth_quadrant_l3294_329451

theorem power_function_not_in_fourth_quadrant :
  ∀ (a : ℝ) (x : ℝ), 
    a ∈ ({1, 2, 3, (1/2 : ℝ), -1} : Set ℝ) → 
    x > 0 → 
    x^a > 0 := by
  sorry

end power_function_not_in_fourth_quadrant_l3294_329451


namespace circle_intersection_range_l3294_329474

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m + 6 = 0

-- Define the condition for intersection with y-axis
def intersects_y_axis (m : ℝ) : Prop :=
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ circle_equation 0 y1 m ∧ circle_equation 0 y2 m

-- Define the condition for points being on the same side of the origin
def same_side_of_origin (y1 y2 : ℝ) : Prop :=
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0)

-- Main theorem
theorem circle_intersection_range (m : ℝ) :
  (intersects_y_axis m ∧ 
   ∀ y1 y2 : ℝ, circle_equation 0 y1 m → circle_equation 0 y2 m → same_side_of_origin y1 y2) →
  -6 < m ∧ m < -5 :=
sorry

end circle_intersection_range_l3294_329474


namespace coin_flip_probability_l3294_329449

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of four coins -/
structure FourCoins :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)

/-- The total number of possible outcomes when flipping four coins -/
def totalOutcomes : ℕ := 16

/-- The number of favorable outcomes (penny heads, nickel heads, dime tails) -/
def favorableOutcomes : ℕ := 2

/-- The probability of the desired outcome -/
def desiredProbability : ℚ := 1 / 8

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = desiredProbability := by
  sorry

end coin_flip_probability_l3294_329449


namespace pencil_cost_proof_l3294_329468

/-- The cost of 4 pencils and 5 pens in dollars -/
def total_cost_1 : ℚ := 2

/-- The cost of 3 pencils and 4 pens in dollars -/
def total_cost_2 : ℚ := 79/50

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

theorem pencil_cost_proof :
  ∃ (pen_cost : ℚ),
    4 * pencil_cost + 5 * pen_cost = total_cost_1 ∧
    3 * pencil_cost + 4 * pen_cost = total_cost_2 :=
by sorry

end pencil_cost_proof_l3294_329468


namespace decimal_to_binary_38_l3294_329421

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end decimal_to_binary_38_l3294_329421


namespace odd_sum_count_l3294_329415

def card_set : Finset ℕ := {1, 2, 3, 4}

def is_sum_odd (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_sum_odd pair)

theorem odd_sum_count : odd_sum_pairs.card = 4 := by
  sorry

end odd_sum_count_l3294_329415


namespace saplings_distribution_l3294_329483

theorem saplings_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2126 →
  a = 2 * b + 20 →
  a = 3 * c + 24 →
  a = 5 * d - 45 →
  a + b + c + d = total →
  a = 1050 := by
  sorry

end saplings_distribution_l3294_329483


namespace negation_of_universal_proposition_l3294_329478

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end negation_of_universal_proposition_l3294_329478


namespace paperClips_in_two_cases_l3294_329410

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 300

/-- Theorem: The number of paper clips in 2 cases is 2c * b * 300 -/
theorem paperClips_in_two_cases (c b : ℕ) : 
  paperClipsIn2Cases c b = 2 * c * b * 300 := by
  sorry

end paperClips_in_two_cases_l3294_329410


namespace card_game_total_l3294_329499

theorem card_game_total (total : ℕ) (ellis orion : ℕ) : 
  ellis = (11 : ℕ) * total / 20 →
  orion = (9 : ℕ) * total / 20 →
  ellis = orion + 50 →
  total = 500 := by
sorry

end card_game_total_l3294_329499


namespace bob_and_alice_heights_l3294_329428

/-- The problem statement about Bob and Alice's heights --/
theorem bob_and_alice_heights :
  ∀ (initial_height : ℝ) (bob_growth_percent : ℝ) (alice_growth_ratio : ℝ) (bob_final_height : ℝ),
  initial_height > 0 →
  bob_growth_percent = 0.25 →
  alice_growth_ratio = 1/3 →
  bob_final_height = 75 →
  bob_final_height = initial_height * (1 + bob_growth_percent) →
  let bob_growth_inches := initial_height * bob_growth_percent
  let alice_growth_inches := bob_growth_inches * alice_growth_ratio
  let alice_final_height := initial_height + alice_growth_inches
  alice_final_height = 65 := by
sorry


end bob_and_alice_heights_l3294_329428


namespace geometric_sequence_sum_l3294_329464

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 1 →
  (∀ n, S n = (a 1 - a (n + 1) * (a 2 / a 1)^n) / (1 - a 2 / a 1)) →
  1 / a 1 - 1 / a 2 = 2 / a 3 →
  S 4 = 15 := by
sorry

end geometric_sequence_sum_l3294_329464
