import Mathlib

namespace dog_area_theorem_l2817_281782

/-- Represents a rectangular obstruction -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the position where the dog is tied -/
structure TiePoint where
  distance_from_midpoint : ℝ

/-- Calculates the area accessible by a dog tied to a point near a rectangular obstruction -/
def accessible_area (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) : ℝ :=
  sorry

/-- Theorem stating the accessible area for the given problem -/
theorem dog_area_theorem (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) :
  rect.length = 20 ∧ rect.width = 10 ∧ tie.distance_from_midpoint = 5 ∧ rope_length = 10 →
  accessible_area rect tie rope_length = 62.5 * Real.pi :=
sorry

end dog_area_theorem_l2817_281782


namespace a_squared_ge_three_l2817_281785

theorem a_squared_ge_three (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = a * b * c) (h3 : a^2 = b * c) : a^2 ≥ 3 := by
  sorry

end a_squared_ge_three_l2817_281785


namespace lingonberries_to_pick_thursday_l2817_281703

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The number of days Steve has to make the money -/
def total_days : ℕ := 4

/-- The amount of money Steve earns per pound of lingonberries -/
def money_per_pound : ℕ := 2

/-- The amount of lingonberries Steve picked on Monday -/
def monday_picked : ℕ := 8

/-- The amount of lingonberries Steve picked on Tuesday relative to Monday -/
def tuesday_multiplier : ℕ := 3

/-- The amount of lingonberries Steve picked on Wednesday -/
def wednesday_picked : ℕ := 0

theorem lingonberries_to_pick_thursday : 
  (total_money / money_per_pound) - 
  (monday_picked + tuesday_multiplier * monday_picked + wednesday_picked) = 18 := by
  sorry

end lingonberries_to_pick_thursday_l2817_281703


namespace trigonometric_identities_l2817_281767

theorem trigonometric_identities (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.cos (θ - π)) / (Real.sin (θ + π) + Real.cos (θ + π)) = -1/3 ∧
  Real.sin (2 * θ) = 4/5 := by
  sorry

end trigonometric_identities_l2817_281767


namespace mod_inverse_of_5_mod_33_l2817_281720

theorem mod_inverse_of_5_mod_33 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 := by
  sorry

end mod_inverse_of_5_mod_33_l2817_281720


namespace students_taking_one_subject_l2817_281740

theorem students_taking_one_subject (both : ℕ) (algebra : ℕ) (geometry_only : ℕ)
  (h1 : both = 16)
  (h2 : algebra = 36)
  (h3 : geometry_only = 15) :
  algebra - both + geometry_only = 35 := by
  sorry

end students_taking_one_subject_l2817_281740


namespace total_vehicles_on_highway_l2817_281700

theorem total_vehicles_on_highway : 
  ∀ (num_trucks : ℕ) (num_cars : ℕ),
  num_trucks = 100 →
  num_cars = 2 * num_trucks →
  num_cars + num_trucks = 300 :=
by
  sorry

end total_vehicles_on_highway_l2817_281700


namespace min_cards_xiaohua_l2817_281775

def greeting_cards (x y z : ℕ) : Prop :=
  (Nat.lcm x (Nat.lcm y z) = 60) ∧
  (Nat.gcd x y = 4) ∧
  (Nat.gcd y z = 3) ∧
  (x ≥ 5)

theorem min_cards_xiaohua :
  ∀ x y z : ℕ, greeting_cards x y z → x ≥ 20 := by
  sorry

end min_cards_xiaohua_l2817_281775


namespace negation_equivalence_l2817_281704

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∀ x : ℝ, x ≤ 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end negation_equivalence_l2817_281704


namespace cannot_form_triangle_l2817_281780

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 2, 4, and 6 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 2 4 6) := by
  sorry

end cannot_form_triangle_l2817_281780


namespace ticket_sales_solution_l2817_281796

/-- Represents the number of tickets sold for each type -/
structure TicketSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the conditions of the ticket sales problem -/
def validTicketSales (s : TicketSales) : Prop :=
  s.a + s.b + s.c = 400 ∧
  50 * s.a + 40 * s.b + 30 * s.c = 15500 ∧
  s.b = s.c

/-- Theorem stating the solution to the ticket sales problem -/
theorem ticket_sales_solution :
  ∃ (s : TicketSales), validTicketSales s ∧ s.a = 100 ∧ s.b = 150 ∧ s.c = 150 := by
  sorry


end ticket_sales_solution_l2817_281796


namespace stock_change_theorem_l2817_281717

/-- The overall percent change in a stock after two days of trading -/
def overall_percent_change (day1_decrease : ℝ) (day2_increase : ℝ) : ℝ :=
  (((1 - day1_decrease) * (1 + day2_increase)) - 1) * 100

/-- Theorem stating the overall percent change for the given scenario -/
theorem stock_change_theorem :
  overall_percent_change 0.25 0.35 = 1.25 := by
  sorry

#eval overall_percent_change 0.25 0.35

end stock_change_theorem_l2817_281717


namespace arithmetic_seq_common_diff_l2817_281727

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If (S_3/3) - (S_2/2) = 1 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_seq_common_diff
  (seq : ArithmeticSequence)
  (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) :
  seq.d = 2 := by
  sorry

end arithmetic_seq_common_diff_l2817_281727


namespace largest_six_digit_number_with_divisibility_l2817_281787

theorem largest_six_digit_number_with_divisibility (A : ℕ) : 
  A ≤ 999999 ∧ 
  A ≥ 100000 ∧
  A % 19 = 0 ∧ 
  (A / 10) % 17 = 0 ∧ 
  (A / 100) % 13 = 0 →
  A ≤ 998412 :=
by sorry

end largest_six_digit_number_with_divisibility_l2817_281787


namespace shelter_dogs_count_l2817_281732

theorem shelter_dogs_count (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 8) = 15 / 11 →
  dogs = 30 := by
sorry

end shelter_dogs_count_l2817_281732


namespace quadratic_minimum_l2817_281771

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 13 ≥ 4 := by
  sorry

end quadratic_minimum_l2817_281771


namespace symmetric_periodic_function_properties_l2817_281765

open Real

/-- A function satisfying specific symmetry and periodicity properties -/
structure SymmetricPeriodicFunction (a c d : ℝ) where
  f : ℝ → ℝ
  even_at_a : ∀ x, f (a + x) = f (a - x)
  sum_at_c : ∀ x, f (c + x) + f (c - x) = 2 * d
  a_neq_c : a ≠ c

theorem symmetric_periodic_function_properties
  {a c d : ℝ} (spf : SymmetricPeriodicFunction a c d) :
  (∀ x, (deriv spf.f) (c + x) = (deriv spf.f) (c - x)) ∧
  (∀ x, spf.f (x + 2 * |c - a|) = 2 * d - spf.f x) ∧
  (∀ x, spf.f (spf.f (a + x)) = spf.f (spf.f (a - x))) :=
by sorry

end symmetric_periodic_function_properties_l2817_281765


namespace sqrt_ratio_simplification_l2817_281762

theorem sqrt_ratio_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (49 + 64)) = 17 / Real.sqrt 113 := by
  sorry

end sqrt_ratio_simplification_l2817_281762


namespace a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l2817_281769

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ a ≤ 1) := by
  sorry

end a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l2817_281769


namespace quadratic_minimum_l2817_281723

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 1 ≥ -8 ∧ ∃ y : ℝ, y^2 + 6*y + 1 = -8 := by
  sorry

end quadratic_minimum_l2817_281723


namespace opposite_of_negative_sqrt_two_l2817_281721

theorem opposite_of_negative_sqrt_two : -((-Real.sqrt 2)) = Real.sqrt 2 := by
  sorry

end opposite_of_negative_sqrt_two_l2817_281721


namespace min_sequence_length_is_eight_l2817_281709

/-- The set S containing elements 1, 2, 3, and 4 -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A sequence of natural numbers -/
def Sequence := List ℕ

/-- Check if a list contains exactly the elements of a given set -/
def containsExactly (l : List ℕ) (s : Finset ℕ) : Prop :=
  l.toFinset = s

/-- Check if a sequence satisfies the property for all non-empty subsets of S -/
def satisfiesProperty (seq : Sequence) : Prop :=
  ∀ B : Finset ℕ, B ⊆ S → B.Nonempty → 
    ∃ subseq : List ℕ, subseq.length = B.card ∧ 
      seq.Sublist subseq ∧ containsExactly subseq B

/-- The minimum length of a sequence satisfying the property -/
def minSequenceLength : ℕ := 8

/-- Theorem stating that the minimum length of a sequence satisfying the property is 8 -/
theorem min_sequence_length_is_eight :
  (∃ seq : Sequence, seq.length = minSequenceLength ∧ satisfiesProperty seq) ∧
  (∀ seq : Sequence, seq.length < minSequenceLength → ¬satisfiesProperty seq) := by
  sorry


end min_sequence_length_is_eight_l2817_281709


namespace sum_of_ages_sum_of_ages_proof_l2817_281706

/-- Proves that the sum of James and Louise's current ages is 32 years. -/
theorem sum_of_ages : ℝ → ℝ → Prop :=
  fun james louise =>
    james = louise + 9 →
    james + 5 = 3 * (louise - 3) →
    james + louise = 32

-- The proof is omitted
theorem sum_of_ages_proof : ∃ (james louise : ℝ), sum_of_ages james louise :=
  sorry

end sum_of_ages_sum_of_ages_proof_l2817_281706


namespace girl_scout_pool_trip_expenses_l2817_281736

/-- Girl Scout Pool Trip Expenses Theorem -/
theorem girl_scout_pool_trip_expenses
  (earnings : ℝ)
  (pool_entry_cost : ℝ)
  (transportation_fee : ℝ)
  (snack_cost : ℝ)
  (num_people : ℕ)
  (h1 : earnings = 30)
  (h2 : pool_entry_cost = 2.5)
  (h3 : transportation_fee = 1.25)
  (h4 : snack_cost = 3)
  (h5 : num_people = 10) :
  earnings - (pool_entry_cost + transportation_fee + snack_cost) * num_people = -37.5 :=
sorry

end girl_scout_pool_trip_expenses_l2817_281736


namespace discount_difference_l2817_281784

theorem discount_difference : 
  let first_discount : ℝ := 0.25
  let second_discount : ℝ := 0.15
  let third_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.45
  let true_discount : ℝ := 1 - (1 - first_discount) * (1 - second_discount) * (1 - third_discount)
  claimed_discount - true_discount = 0.02375 := by
sorry

end discount_difference_l2817_281784


namespace meaningful_expression_l2817_281772

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (4 - x)) ↔ x < 4 :=
by sorry

end meaningful_expression_l2817_281772


namespace work_completion_time_l2817_281719

theorem work_completion_time 
  (total_pay : ℝ) 
  (a_time : ℝ) 
  (b_time : ℝ) 
  (c_pay : ℝ) : 
  total_pay = 500 →
  a_time = 5 →
  b_time = 10 →
  c_pay = 200 →
  ∃ (completion_time : ℝ),
    completion_time = 2 ∧
    (1 / completion_time) = (1 / a_time) + (1 / b_time) + (c_pay / total_pay) / completion_time :=
by sorry

end work_completion_time_l2817_281719


namespace wall_bricks_l2817_281742

/-- Represents the time taken by Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time taken by Jerry to build the wall alone -/
def jerry_time : ℝ := 8

/-- Represents the decrease in combined output when working together -/
def output_decrease : ℝ := 15

/-- Represents the time taken to complete the job together -/
def combined_time : ℝ := 6

/-- Theorem stating that the number of bricks in the wall is 240 -/
theorem wall_bricks : ℝ := by
  sorry

end wall_bricks_l2817_281742


namespace square_root_division_l2817_281729

theorem square_root_division (x : ℝ) : (Real.sqrt 5776 / x = 4) → x = 19 := by
  sorry

end square_root_division_l2817_281729


namespace jonathan_tax_calculation_l2817_281713

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxInCents (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.4%, the local tax amount is 60 cents. -/
theorem jonathan_tax_calculation :
  localTaxInCents 25 2.4 = 60 := by
  sorry

#eval localTaxInCents 25 2.4

end jonathan_tax_calculation_l2817_281713


namespace tan_45_degrees_l2817_281731

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l2817_281731


namespace apple_cost_l2817_281724

/-- Proves that the cost of each apple is 4 dollars given the conditions -/
theorem apple_cost (total_money : ℕ) (kids : ℕ) (apples_per_kid : ℕ) :
  total_money = 360 →
  kids = 18 →
  apples_per_kid = 5 →
  total_money / (kids * apples_per_kid) = 4 := by
sorry

end apple_cost_l2817_281724


namespace no_digit_reversal_double_l2817_281754

theorem no_digit_reversal_double :
  (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → 10 * b + a ≠ 2 * (10 * a + b)) ∧
  (∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    100 * c + 10 * b + a ≠ 2 * (100 * a + 10 * b + c)) := by
  sorry

end no_digit_reversal_double_l2817_281754


namespace right_triangle_equation_roots_l2817_281799

theorem right_triangle_equation_roots (a b c : ℝ) (h_right_angle : a^2 + c^2 = b^2) :
  ∃ (x : ℝ), ¬ (∀ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ↔ x = y) ∧
             ¬ (∀ (y z : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ∧
                             a * (z^2 - 1) - 2 * z + b * (z^2 + 1) = 0 → y ≠ z) ∧
             ¬ (¬ ∃ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0) :=
by sorry

end right_triangle_equation_roots_l2817_281799


namespace cost_per_load_is_25_cents_l2817_281755

/-- Calculates the cost per load in cents when buying detergent on sale -/
def cost_per_load_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) : ℚ :=
  let total_cost := 2 * sale_price_per_bottle
  let total_loads := 2 * loads_per_bottle
  (total_cost / total_loads) * 100

/-- Theorem stating that the cost per load is 25 cents under given conditions -/
theorem cost_per_load_is_25_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) 
    (h1 : loads_per_bottle = 80)
    (h2 : sale_price_per_bottle = 20) :
  cost_per_load_cents loads_per_bottle sale_price_per_bottle = 25 := by
  sorry

end cost_per_load_is_25_cents_l2817_281755


namespace f_bounds_in_R_f_attains_bounds_l2817_281716

/-- The triangular region R with vertices A(4,1), B(-1,-6), C(-3,2) -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (4*a - b - 3*c, a - 6*b + 2*c)}

/-- The function to be maximized and minimized -/
def f (p : ℝ × ℝ) : ℝ := 4 * p.1 - 3 * p.2

theorem f_bounds_in_R :
  ∀ p ∈ R, -18 ≤ f p ∧ f p ≤ 14 :=
by sorry

theorem f_attains_bounds :
  (∃ p ∈ R, f p = -18) ∧ (∃ p ∈ R, f p = 14) :=
by sorry

end f_bounds_in_R_f_attains_bounds_l2817_281716


namespace point_C_coordinates_l2817_281712

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (0, 5)

def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
    parallel (vector A C) (vector O A) →
    perpendicular (vector B C) (vector A B) →
    C = (12, -4) := by sorry

end point_C_coordinates_l2817_281712


namespace tens_digit_of_13_pow_2023_l2817_281779

theorem tens_digit_of_13_pow_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] ∧ n < 10 :=
by sorry

end tens_digit_of_13_pow_2023_l2817_281779


namespace percentage_excess_l2817_281797

theorem percentage_excess (x y : ℝ) (h : x = 0.8 * y) : y = 1.25 * x := by
  sorry

end percentage_excess_l2817_281797


namespace rectangular_plot_poles_l2817_281794

/-- Calculate the number of poles needed for a rectangular fence --/
def poles_needed (length width long_spacing short_spacing : ℕ) : ℕ :=
  let long_poles := (length / long_spacing + 1) * 2
  let short_poles := (width / short_spacing + 1) * 2
  long_poles + short_poles - 4

/-- Theorem: The number of poles needed for the given rectangular plot is 70 --/
theorem rectangular_plot_poles :
  poles_needed 90 70 4 5 = 70 := by
  sorry

end rectangular_plot_poles_l2817_281794


namespace fuchsia_to_mauve_l2817_281722

/-- Represents the composition of paint mixtures -/
structure PaintMix where
  red : ℚ
  blue : ℚ

/-- The ratio of red to blue in fuchsia paint -/
def fuchsia : PaintMix := { red := 6, blue := 3 }

/-- The ratio of red to blue in mauve paint -/
def mauve : PaintMix := { red := 4, blue := 5 }

/-- The amount of blue paint needed to change fuchsia to mauve -/
def blue_paint_needed (F : ℚ) : ℚ := F / 2

theorem fuchsia_to_mauve (F : ℚ) (F_pos : F > 0) :
  let original_red := F * fuchsia.red / (fuchsia.red + fuchsia.blue)
  let original_blue := F * fuchsia.blue / (fuchsia.red + fuchsia.blue)
  let added_blue := blue_paint_needed F
  original_red / (original_blue + added_blue) = mauve.red / mauve.blue :=
by sorry

end fuchsia_to_mauve_l2817_281722


namespace total_subjects_l2817_281792

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 79)
  (h2 : average_five = 74)
  (h3 : last_subject = 104) : 
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + last_subject :=
by sorry

end total_subjects_l2817_281792


namespace sum_even_integers_minus15_to_5_l2817_281759

def sum_even_integers (a b : Int) : Int :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let num_terms := (last_even - first_even) / 2 + 1
  (first_even + last_even) * num_terms / 2

theorem sum_even_integers_minus15_to_5 :
  sum_even_integers (-15) 5 = -50 := by
sorry

end sum_even_integers_minus15_to_5_l2817_281759


namespace round_table_seats_l2817_281726

/-- A round table with equally spaced seats numbered clockwise. -/
structure RoundTable where
  num_seats : ℕ
  seat_numbers : Fin num_seats → ℕ
  seat_numbers_clockwise : ∀ (i j : Fin num_seats), i < j → seat_numbers i < seat_numbers j

/-- Two seats are opposite if they are half the total number of seats apart. -/
def are_opposite (t : RoundTable) (s1 s2 : Fin t.num_seats) : Prop :=
  (s2.val + t.num_seats / 2) % t.num_seats = s1.val

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.num_seats) :
  t.seat_numbers s1 = 10 →
  t.seat_numbers s2 = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end round_table_seats_l2817_281726


namespace pairwise_sums_problem_l2817_281776

theorem pairwise_sums_problem (a b c d e x y : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {183, 186, 187, 190, 191, 192, 193, 194, 196, x} ∧
  x > 196 ∧
  y = 10 * x + 3 →
  a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200 ∧ y = 2003 :=
by sorry

end pairwise_sums_problem_l2817_281776


namespace floor_equation_natural_numbers_l2817_281774

theorem floor_equation_natural_numbers (a b : ℕ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (b = a^2 + 1 ∨ a = b^2 + 1) :=
sorry

end floor_equation_natural_numbers_l2817_281774


namespace equilateral_triangle_perimeter_l2817_281711

/-- The sum of all sides of an equilateral triangle with side length 13/12 meters is 13/4 meters. -/
theorem equilateral_triangle_perimeter (side_length : ℚ) (h : side_length = 13 / 12) :
  3 * side_length = 13 / 4 := by
  sorry

end equilateral_triangle_perimeter_l2817_281711


namespace students_per_bus_l2817_281753

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (students_in_cars : ℕ) :
  total_students = 375 →
  num_buses = 7 →
  students_in_cars = 4 →
  (total_students - students_in_cars) / num_buses = 53 := by
  sorry

end students_per_bus_l2817_281753


namespace min_value_quadratic_min_value_is_two_min_value_achieved_l2817_281788

theorem min_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + a*b + 3*b^2 = 10 → x^2 - x*y + y^2 ≤ a^2 - a*b + b^2 :=
by sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  x^2 - x*y + y^2 ≥ 2 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + x*y + 3*y^2 = 10 ∧ x^2 - x*y + y^2 < 2 + ε :=
by sorry

end min_value_quadratic_min_value_is_two_min_value_achieved_l2817_281788


namespace chocolate_problem_l2817_281752

theorem chocolate_problem (C S : ℝ) (N : ℕ) :
  (N * C = 77 * S) →  -- The total cost price equals the selling price of 77 chocolates
  ((S - C) / C = 4 / 7) →  -- The gain percent is 4/7
  N = 121 := by  -- Prove that N (number of chocolates bought at cost price) is 121
sorry

end chocolate_problem_l2817_281752


namespace smallest_positive_multiple_of_32_l2817_281728

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * n ≥ 32 := by
  sorry

end smallest_positive_multiple_of_32_l2817_281728


namespace calculation_difference_l2817_281730

def correct_calculation : ℤ := 12 - (3 * 4 + 2)

def incorrect_calculation : ℤ := 12 - 3 * 4 + 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -4 := by
  sorry

end calculation_difference_l2817_281730


namespace range_of_b_minus_a_l2817_281702

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end range_of_b_minus_a_l2817_281702


namespace art_gallery_sculpture_fraction_l2817_281747

theorem art_gallery_sculpture_fraction 
  (total : ℕ) 
  (displayed : ℕ) 
  (sculptures_not_displayed : ℕ) 
  (h1 : displayed = total / 3)
  (h2 : sculptures_not_displayed = 800)
  (h3 : total = 1800)
  (h4 : (total - displayed) / 3 = total - displayed - sculptures_not_displayed) :
  3 * (sculptures_not_displayed + displayed - total + sculptures_not_displayed) = 2 * displayed := by
  sorry

end art_gallery_sculpture_fraction_l2817_281747


namespace car_payment_remainder_l2817_281751

theorem car_payment_remainder (part_payment : ℝ) (percentage : ℝ) (total_cost : ℝ) (remainder : ℝ) : 
  part_payment = 300 →
  percentage = 5 →
  part_payment = percentage / 100 * total_cost →
  remainder = total_cost - part_payment →
  remainder = 5700 := by
sorry

end car_payment_remainder_l2817_281751


namespace inequality_proof_l2817_281766

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end inequality_proof_l2817_281766


namespace circle_distance_extrema_l2817_281773

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≥ d Q) ∧
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≤ d Q) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≤ 74) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≥ 34) :=
sorry

end circle_distance_extrema_l2817_281773


namespace translation_result_l2817_281746

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally by a given distance -/
def translate_x (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- The initial point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- The translation distance to the right -/
def translation_distance : ℝ := 1

theorem translation_result :
  translate_x P translation_distance = { x := -1, y := 4 } := by
  sorry

end translation_result_l2817_281746


namespace pipe_fill_time_l2817_281783

/-- The time (in hours) it takes for Pipe A to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 8

/-- The time (in hours) it takes for Pipe A to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 12

/-- The time (in hours) it takes for the leak to empty the full tank -/
def empty_time : ℝ := 24

theorem pipe_fill_time :
  (1 / fill_time_without_leak) - (1 / empty_time) = (1 / fill_time_with_leak) :=
sorry

end pipe_fill_time_l2817_281783


namespace perfect_games_count_l2817_281714

theorem perfect_games_count (perfect_score : ℕ) (total_points : ℕ) : 
  perfect_score = 21 → total_points = 63 → total_points / perfect_score = 3 := by
sorry

end perfect_games_count_l2817_281714


namespace min_max_values_l2817_281798

theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / a + 1 / b ≥ 9) ∧ (a * b ≤ 1 / 16) := by sorry

end min_max_values_l2817_281798


namespace hexagon_interior_angles_sum_l2817_281715

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720° -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end hexagon_interior_angles_sum_l2817_281715


namespace water_bottle_shape_l2817_281757

/-- Represents the volume of water in a bottle as a function of height -/
noncomputable def VolumeFunction := ℝ → ℝ

/-- A water bottle with a given height and volume function -/
structure WaterBottle where
  height : ℝ
  volume : VolumeFunction
  height_pos : height > 0

/-- The shape of a water bottle is non-linear and increases faster than linear growth -/
def IsNonLinearIncreasing (b : WaterBottle) : Prop :=
  b.volume (b.height / 2) > (1 / 2) * b.volume b.height

theorem water_bottle_shape (b : WaterBottle) 
  (h : IsNonLinearIncreasing b) : 
  ∃ (k : ℝ), k > 0 ∧ ∀ h, 0 ≤ h ∧ h ≤ b.height → b.volume h = k * h^2 :=
sorry

end water_bottle_shape_l2817_281757


namespace polynomial_independence_l2817_281745

theorem polynomial_independence (x m : ℝ) : 
  (∀ m, 6 * x^2 + (1 - 2*m) * x + 7*m = 6 * x^2 + x) → x = 7/2 := by
  sorry

end polynomial_independence_l2817_281745


namespace equation_value_l2817_281737

theorem equation_value : 
  let Y : ℝ := (180 * 0.15 - (180 * 0.15) / 3) + 0.245 * (2 / 3 * 270) - (5.4 * 2) / (0.25^2)
  Y = -110.7 := by
  sorry

end equation_value_l2817_281737


namespace binomial_unique_solution_l2817_281701

/-- Represents a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem stating the unique solution for n and p given E(ξ) and D(ξ) -/
theorem binomial_unique_solution :
  ∀ ξ : BinomialDistribution,
    expectation ξ = 12 →
    variance ξ = 4 →
    ξ.n = 18 ∧ ξ.p = 2/3 :=
by sorry

end binomial_unique_solution_l2817_281701


namespace isosceles_right_triangle_hypotenuse_l2817_281725

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : a = 10) :
  let hypotenuse := a * Real.sqrt 2
  hypotenuse = 10 * Real.sqrt 2 := by sorry

end isosceles_right_triangle_hypotenuse_l2817_281725


namespace circle_largest_area_l2817_281760

-- Define the shapes
def triangle_area (side : Real) (angle1 : Real) (angle2 : Real) : Real :=
  -- Area calculation for triangle
  sorry

def rhombus_area (d1 : Real) (d2 : Real) (angle : Real) : Real :=
  -- Area calculation for rhombus
  sorry

def circle_area (radius : Real) : Real :=
  -- Area calculation for circle
  sorry

def square_area (diagonal : Real) : Real :=
  -- Area calculation for square
  sorry

-- Theorem statement
theorem circle_largest_area :
  let triangle_a := triangle_area (Real.sqrt 2) (60 * π / 180) (45 * π / 180)
  let rhombus_a := rhombus_area (Real.sqrt 2) (Real.sqrt 3) (75 * π / 180)
  let circle_a := circle_area 1
  let square_a := square_area 2.5
  circle_a > triangle_a ∧ circle_a > rhombus_a ∧ circle_a > square_a :=
by sorry


end circle_largest_area_l2817_281760


namespace rental_miles_driven_l2817_281795

-- Define the rental parameters
def daily_rate : ℚ := 29
def mile_rate : ℚ := 0.08
def total_paid : ℚ := 46.12

-- Define the function to calculate miles driven
def miles_driven (daily_rate mile_rate total_paid : ℚ) : ℚ :=
  (total_paid - daily_rate) / mile_rate

-- Theorem statement
theorem rental_miles_driven :
  miles_driven daily_rate mile_rate total_paid = 214 := by
  sorry

end rental_miles_driven_l2817_281795


namespace third_candidate_votes_l2817_281770

theorem third_candidate_votes :
  ∀ (total_votes : ℕ) (winning_votes second_votes third_votes : ℕ),
    winning_votes = 11628 →
    second_votes = 7636 →
    winning_votes = (49.69230769230769 / 100 : ℚ) * total_votes →
    total_votes = winning_votes + second_votes + third_votes →
    third_votes = 4136 := by
  sorry

end third_candidate_votes_l2817_281770


namespace grocery_problem_l2817_281739

theorem grocery_problem (total_packs : ℕ) (cookie_packs : ℕ) (noodle_packs : ℕ) :
  total_packs = 28 →
  cookie_packs = 12 →
  total_packs = cookie_packs + noodle_packs →
  noodle_packs = 16 := by
sorry

end grocery_problem_l2817_281739


namespace divisor_problem_l2817_281791

theorem divisor_problem (n d : ℕ) : 
  (n % d = 3) → (n^2 % d = 3) → d = 6 := by sorry

end divisor_problem_l2817_281791


namespace fraction_simplification_l2817_281764

theorem fraction_simplification (x : ℝ) : (x + 3) / 4 - (5 - 2*x) / 3 = (11*x - 11) / 12 := by
  sorry

end fraction_simplification_l2817_281764


namespace rajesh_savings_percentage_l2817_281718

theorem rajesh_savings_percentage (monthly_salary : ℝ) (food_percentage : ℝ) (medicine_percentage : ℝ) (savings : ℝ) : 
  monthly_salary = 15000 →
  food_percentage = 40 →
  medicine_percentage = 20 →
  savings = 4320 →
  let remaining := monthly_salary - (food_percentage / 100 * monthly_salary) - (medicine_percentage / 100 * monthly_salary)
  (savings / remaining) * 100 = 72 := by
sorry

end rajesh_savings_percentage_l2817_281718


namespace arithmetic_mean_squares_first_four_odd_numbers_l2817_281748

theorem arithmetic_mean_squares_first_four_odd_numbers : 
  let odd_numbers := [1, 3, 5, 7]
  let squares := List.map (λ x => x^2) odd_numbers
  (List.sum squares) / (List.length squares) = 21 := by
  sorry

end arithmetic_mean_squares_first_four_odd_numbers_l2817_281748


namespace dave_final_tickets_l2817_281758

def arcade_tickets (initial_tickets : ℕ) (candy_cost : ℕ) (beanie_cost : ℕ) (racing_game_win : ℕ) : ℕ :=
  let remaining_tickets := initial_tickets - (candy_cost + beanie_cost)
  let tickets_before_challenge := remaining_tickets + racing_game_win
  2 * tickets_before_challenge

theorem dave_final_tickets :
  arcade_tickets 11 3 5 10 = 26 := by
  sorry

end dave_final_tickets_l2817_281758


namespace log_inequality_l2817_281708

theorem log_inequality (x y : ℝ) (h : Real.log x < Real.log y ∧ Real.log y < 0) : 
  0 < x ∧ x < y ∧ y < 1 := by sorry

end log_inequality_l2817_281708


namespace union_of_sets_l2817_281763

/-- Given sets A and B with specific properties, prove their union -/
theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {|a + 1|, 3, 5}
  let B : Set ℝ := {2*a + 1, a^(2*a + 2), a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {1, 2, 3, 5}) := by
  sorry


end union_of_sets_l2817_281763


namespace both_are_dwarves_l2817_281768

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| GoldStatement : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (stmnt : Statement) : Prop :=
  match speaker, stmnt with
  | Inhabitant.Dwarf, Statement.GoldStatement => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- A's statement
def a_statement : Statement := Statement.GoldStatement

-- B's statement about A
def b_statement (a_type : Inhabitant) : Statement :=
  match a_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (a_type b_type : Inhabitant),
    a_type = Inhabitant.Dwarf ∧
    b_type = Inhabitant.Dwarf ∧
    isTruthful a_type a_statement = False ∧
    isTruthful b_type (b_statement a_type) = True :=
sorry

end both_are_dwarves_l2817_281768


namespace wendy_distance_difference_l2817_281777

/-- The distance Wendy ran in miles -/
def distance_run : ℝ := 19.83

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The difference between the distance Wendy ran and walked -/
def distance_difference : ℝ := distance_run - distance_walked

theorem wendy_distance_difference :
  distance_difference = 10.66 := by sorry

end wendy_distance_difference_l2817_281777


namespace fraction_division_l2817_281743

theorem fraction_division (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end fraction_division_l2817_281743


namespace min_value_trig_expression_l2817_281781

theorem min_value_trig_expression : 
  ∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (9/10) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end min_value_trig_expression_l2817_281781


namespace weight_of_e_l2817_281786

/-- Given three weights d, e, and f, prove that e equals 82 when their average is 42,
    the average of d and e is 35, and the average of e and f is 41. -/
theorem weight_of_e (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : (e + f) / 2 = 41) : 
  e = 82 := by
  sorry

#check weight_of_e

end weight_of_e_l2817_281786


namespace focus_of_our_parabola_l2817_281744

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax² + bx + c -/
  equation : ℝ → ℝ → Prop

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : Point :=
  sorry

/-- The parabola defined by x² = y -/
def our_parabola : Parabola :=
  { equation := fun x y ↦ x^2 = y }

/-- Theorem stating that the focus of our parabola is at (0, 1) -/
theorem focus_of_our_parabola :
  focus our_parabola = Point.mk 0 1 := by
  sorry

end focus_of_our_parabola_l2817_281744


namespace black_squares_eaten_l2817_281778

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Defines whether a square is black -/
def isBlack (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- The list of squares eaten by termites -/
def eatenSquares : List Square := [
  ⟨3, 1⟩, ⟨4, 6⟩, ⟨3, 7⟩,
  ⟨4, 1⟩, ⟨2, 3⟩, ⟨2, 4⟩, ⟨4, 3⟩,
  ⟨3, 5⟩, ⟨3, 2⟩, ⟨4, 7⟩,
  ⟨3, 6⟩, ⟨2, 6⟩
]

/-- Counts the number of black squares in a list of squares -/
def countBlackSquares (squares : List Square) : Nat :=
  squares.filter isBlack |>.length

/-- Theorem stating that the number of black squares eaten is 12 -/
theorem black_squares_eaten :
  countBlackSquares eatenSquares = 12 := by
  sorry


end black_squares_eaten_l2817_281778


namespace quotient_in_fourth_quadrant_l2817_281735

/-- Given two complex numbers z₁ and z₂, prove that their quotient lies in the fourth quadrant. -/
theorem quotient_in_fourth_quadrant (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 + Complex.I) 
  (hz₂ : z₂ = 1 + Complex.I) : 
  let q := z₁ / z₂
  0 < q.re ∧ q.im < 0 :=
by sorry


end quotient_in_fourth_quadrant_l2817_281735


namespace leading_coefficient_is_negative_fourteen_l2817_281749

def polynomial (x : ℝ) : ℝ := -5 * (x^5 - x^4 + 2*x) + 9 * (x^5 + 3) - 6 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_fourteen :
  ∃ (a : ℝ) (p : ℝ → ℝ), (∀ x, polynomial x = a * x^5 + p x) ∧ (∀ x, x ≠ 0 → |p x| / |x|^5 < 1) ∧ a = -14 :=
sorry

end leading_coefficient_is_negative_fourteen_l2817_281749


namespace min_k_for_A_cannot_win_l2817_281761

/-- Represents a position on the infinite hexagonal grid --/
structure HexPosition

/-- Represents the game state --/
structure GameState where
  board : HexPosition → Option Bool  -- True for A's counter, False for empty
  turn : Bool  -- True for A's turn, False for B's turn

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : HexPosition) : Prop := sorry

/-- Checks if there are k consecutive counters in a line --/
def consecutive_counters (state : GameState) (k : ℕ) : Prop := sorry

/-- Represents a valid move by player A --/
def valid_move_A (state : GameState) (p1 p2 : HexPosition) : Prop :=
  adjacent p1 p2 ∧ state.board p1 = none ∧ state.board p2 = none ∧ state.turn

/-- Represents a valid move by player B --/
def valid_move_B (state : GameState) (p : HexPosition) : Prop :=
  state.board p = some true ∧ ¬state.turn

/-- The main theorem stating that 6 is the minimum k for which A cannot win --/
theorem min_k_for_A_cannot_win :
  (∀ k < 6, ∃ (strategy : GameState → HexPosition × HexPosition),
    ∀ (counter_strategy : GameState → HexPosition),
      ∃ (n : ℕ), ∃ (final_state : GameState),
        consecutive_counters final_state k) ∧
  (∀ (strategy : GameState → HexPosition × HexPosition),
    ∃ (counter_strategy : GameState → HexPosition),
      ∀ (n : ℕ), ∀ (final_state : GameState),
        ¬consecutive_counters final_state 6) :=
sorry

end min_k_for_A_cannot_win_l2817_281761


namespace arithmetic_sequence_properties_l2817_281707

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_properties :
  let a₁ : ℚ := 4
  let d : ℚ := 5
  let seq := arithmetic_sequence a₁ d
  (seq 3 * seq 6 = 406) ∧
  (∃ (q r : ℚ), seq 9 = seq 4 * q + r ∧ q = 2 ∧ r = 6) :=
by sorry

end arithmetic_sequence_properties_l2817_281707


namespace sqrt_sum_equality_l2817_281741

theorem sqrt_sum_equality : 
  Real.sqrt 2 + Real.sqrt (2 + 4) + Real.sqrt (2 + 4 + 6) + Real.sqrt (2 + 4 + 6 + 8) = 
  Real.sqrt 2 + Real.sqrt 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 := by
  sorry

end sqrt_sum_equality_l2817_281741


namespace set_B_determination_l2817_281734

theorem set_B_determination (U A B : Set ℕ) : 
  U = A ∪ B ∧ 
  U = {x : ℕ | 0 ≤ x ∧ x ≤ 10} ∧ 
  A ∩ (U \ B) = {1, 3, 5, 7} → 
  B = {0, 2, 4, 6, 8, 9, 10} := by
  sorry

end set_B_determination_l2817_281734


namespace no_primes_divisible_by_57_l2817_281789

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- State the theorem
theorem no_primes_divisible_by_57 :
  ¬∃ p : ℕ, isPrime p ∧ divides 57 p :=
sorry

end no_primes_divisible_by_57_l2817_281789


namespace total_snakes_l2817_281733

theorem total_snakes (boa_constrictors python rattlesnakes : ℕ) : 
  boa_constrictors = 40 →
  python = 3 * boa_constrictors →
  rattlesnakes = 40 →
  boa_constrictors + python + rattlesnakes = 200 := by
  sorry

end total_snakes_l2817_281733


namespace semicircle_area_comparison_l2817_281705

theorem semicircle_area_comparison : 
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let small_semicircle_radius : ℝ := rectangle_width / 2
  let large_semicircle_radius : ℝ := rectangle_length / 2
  let small_semicircle_area : ℝ := π * small_semicircle_radius^2 / 2
  let large_semicircle_area : ℝ := π * large_semicircle_radius^2 / 2
  (large_semicircle_area / small_semicircle_area - 1) * 100 = 125 := by
sorry

end semicircle_area_comparison_l2817_281705


namespace set_intersection_equality_l2817_281756

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_equality : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end set_intersection_equality_l2817_281756


namespace swimming_contest_outcomes_l2817_281710

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of participants in the swimming contest -/
def num_participants : ℕ := 6

/-- The number of places we're interested in (1st, 2nd, 3rd) -/
def num_places : ℕ := 3

theorem swimming_contest_outcomes :
  permutations num_participants num_places = 120 := by sorry

end swimming_contest_outcomes_l2817_281710


namespace arithmetic_mean_problem_l2817_281790

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 16 + (3*x + 6)) / 5 = 30 → x = 14 := by
sorry

end arithmetic_mean_problem_l2817_281790


namespace arithmetic_mean_problem_l2817_281750

theorem arithmetic_mean_problem (x : ℝ) : 
  (8 + 16 + 21 + 7 + x) / 5 = 12 → x = 8 := by
sorry

end arithmetic_mean_problem_l2817_281750


namespace smallest_n_for_inequality_l2817_281793

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ (1 - 1 / (2^n : ℚ) > 315 / 412) ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 1 - 1 / (2^m : ℚ) ≤ 315 / 412 :=
by sorry

end smallest_n_for_inequality_l2817_281793


namespace white_surface_area_fraction_l2817_281738

theorem white_surface_area_fraction (cube_edge : ℕ) (total_cubes : ℕ) (white_cubes : ℕ) (black_cubes : ℕ) : 
  cube_edge = 4 →
  total_cubes = 64 →
  white_cubes = 48 →
  black_cubes = 16 →
  (cube_edge : ℚ) * (cube_edge : ℚ) * 6 / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) - 
  ((3 * 8 + cube_edge * cube_edge) : ℚ) / ((cube_edge : ℚ) * (cube_edge : ℚ) * 6) = 7 / 12 := by
  sorry

end white_surface_area_fraction_l2817_281738
