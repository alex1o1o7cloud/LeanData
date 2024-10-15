import Mathlib

namespace NUMINAMATH_CALUDE_food_lasts_fifty_days_l2408_240802

/-- The number of days dog food will last given the number of dogs, meals per day, 
    food per meal, number of sacks, and weight per sack. -/
def days_food_lasts (num_dogs : ℕ) (meals_per_day : ℕ) (food_per_meal : ℕ) 
                    (num_sacks : ℕ) (weight_per_sack : ℕ) : ℕ :=
  (num_sacks * weight_per_sack * 1000) / (num_dogs * meals_per_day * food_per_meal)

/-- Proof that given the specific conditions, the food will last 50 days. -/
theorem food_lasts_fifty_days : 
  days_food_lasts 4 2 250 2 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_food_lasts_fifty_days_l2408_240802


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_CaOH2_l2408_240885

/-- Atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Number of Calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- Number of Oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- Number of Hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- Number of moles of Ca(OH)2 -/
def num_moles : ℝ := 3

/-- Molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_3_moles_CaOH2 :
  num_moles * molecular_weight_CaOH2 = 222.30 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_CaOH2_l2408_240885


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l2408_240864

theorem whole_number_between_bounds (N : ℕ) (h : 7.5 < (N : ℝ) / 3 ∧ (N : ℝ) / 3 < 8) : N = 23 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l2408_240864


namespace NUMINAMATH_CALUDE_cookies_per_person_l2408_240838

theorem cookies_per_person (batches : ℕ) (dozen_per_batch : ℕ) (people : ℕ) :
  batches = 4 →
  dozen_per_batch = 2 →
  people = 16 →
  (batches * dozen_per_batch * 12) / people = 6 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2408_240838


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2408_240817

theorem quadratic_factorization (C D : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) →
  C * D + C = 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2408_240817


namespace NUMINAMATH_CALUDE_AF₂_length_l2408_240829

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 27 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define point A on the hyperbola
def A : ℝ × ℝ := sorry
axiom A_on_hyperbola : hyperbola A.1 A.2

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Define AM as the angle bisector
axiom AM_bisector : sorry

-- Theorem to prove
theorem AF₂_length : ‖A - F₂‖ = 6 := by sorry

end NUMINAMATH_CALUDE_AF₂_length_l2408_240829


namespace NUMINAMATH_CALUDE_rope_length_proof_l2408_240867

theorem rope_length_proof (r : ℝ) : 
  r > 0 → 
  π * 20^2 - π * r^2 = 942.8571428571429 → 
  r = 10 := by
sorry

end NUMINAMATH_CALUDE_rope_length_proof_l2408_240867


namespace NUMINAMATH_CALUDE_cafeteria_pies_l2408_240837

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : handed_out = 8)
  (h3 : apples_per_pie = 9) :
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l2408_240837


namespace NUMINAMATH_CALUDE_freds_allowance_l2408_240894

/-- Proves that Fred's weekly allowance is 16 dollars given the problem conditions -/
theorem freds_allowance (spent_on_movies : ℝ) (car_wash_earnings : ℝ) (final_amount : ℝ) :
  spent_on_movies = car_wash_earnings - 6 →
  final_amount = 14 →
  spent_on_movies = 8 →
  spent_on_movies * 2 = 16 :=
by
  sorry

#check freds_allowance

end NUMINAMATH_CALUDE_freds_allowance_l2408_240894


namespace NUMINAMATH_CALUDE_jimin_weight_l2408_240832

theorem jimin_weight (T J : ℝ) (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 := by
  sorry

end NUMINAMATH_CALUDE_jimin_weight_l2408_240832


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2408_240801

/-- The symmetric point of M(2, -3, 1) with respect to the origin is (-2, 3, -1). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ × ℝ := (2, -3, 1)
  let symmetric_point : ℝ × ℝ × ℝ := (-2, 3, -1)
  ∀ (x y z : ℝ), (x, y, z) = M → (-x, -y, -z) = symmetric_point :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2408_240801


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2408_240860

def given_number : ℕ := 7844213
def prime_set : List ℕ := [549, 659, 761]
def result : ℕ := 266866776

theorem smallest_addition_for_divisibility :
  (∀ p ∈ prime_set, (given_number + result) % p = 0) ∧
  (∀ n : ℕ, n < result → ∃ p ∈ prime_set, (given_number + n) % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2408_240860


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2408_240830

theorem book_selection_theorem (total_books : ℕ) (books_to_select : ℕ) (specific_book : ℕ) :
  total_books = 8 →
  books_to_select = 5 →
  specific_book = 1 →
  Nat.choose (total_books - specific_book) (books_to_select - specific_book) = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2408_240830


namespace NUMINAMATH_CALUDE_hexagon_unit_triangles_l2408_240824

/-- The number of unit equilateral triangles in a regular hexagon -/
def num_unit_triangles_in_hexagon (side_length : ℕ) : ℕ :=
  6 * side_length^2

/-- Theorem: A regular hexagon with side length 5 contains 150 unit equilateral triangles -/
theorem hexagon_unit_triangles :
  num_unit_triangles_in_hexagon 5 = 150 := by
  sorry

#eval num_unit_triangles_in_hexagon 5

end NUMINAMATH_CALUDE_hexagon_unit_triangles_l2408_240824


namespace NUMINAMATH_CALUDE_book_profit_percentage_l2408_240843

/-- Calculates the profit percentage given purchase and selling prices in different currencies and their conversion rates to a common currency. -/
def profit_percentage (purchase_price_A : ℚ) (selling_price_B : ℚ) (rate_A_to_C : ℚ) (rate_B_to_C : ℚ) : ℚ :=
  let purchase_price_C := purchase_price_A * rate_A_to_C
  let selling_price_C := selling_price_B * rate_B_to_C
  let profit_C := selling_price_C - purchase_price_C
  (profit_C / purchase_price_C) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 700/3%. -/
theorem book_profit_percentage :
  profit_percentage 50 100 (3/4) (5/4) = 700/3 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l2408_240843


namespace NUMINAMATH_CALUDE_gas_price_calculation_l2408_240878

theorem gas_price_calculation (expected_cash : ℝ) : 
  (12 * (expected_cash / 12) = 10 * (expected_cash / 12 + 0.3)) →
  expected_cash / 12 + 0.3 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_gas_price_calculation_l2408_240878


namespace NUMINAMATH_CALUDE_geometric_sum_base_case_l2408_240876

theorem geometric_sum_base_case (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^2) / (1 - a) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_base_case_l2408_240876


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2408_240851

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  z.re = 0 ∧ z.im ≠ 0 → m = -1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2408_240851


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2408_240809

/-- Pie-eating contest problem -/
theorem pie_eating_contest (bill : ℕ) (adam sierra taylor total : ℕ) : 
  adam = bill + 3 →
  sierra = 2 * bill →
  sierra = 12 →
  taylor = (adam + bill + sierra) / 3 →
  total = adam + bill + sierra + taylor →
  total = 36 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2408_240809


namespace NUMINAMATH_CALUDE_power_5_2048_mod_17_l2408_240875

theorem power_5_2048_mod_17 : 5^2048 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_5_2048_mod_17_l2408_240875


namespace NUMINAMATH_CALUDE_marble_probability_l2408_240889

theorem marble_probability (total red blue : ℕ) (h1 : total = 20) (h2 : red = 7) (h3 : blue = 5) :
  let white := total - (red + blue)
  (red + white : ℚ) / total = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l2408_240889


namespace NUMINAMATH_CALUDE_clean_room_together_l2408_240874

/-- The time it takes for Lisa and Kay to clean their room together -/
theorem clean_room_together (lisa_rate kay_rate : ℝ) (h1 : lisa_rate = 1 / 8) (h2 : kay_rate = 1 / 12) :
  1 / (lisa_rate + kay_rate) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_clean_room_together_l2408_240874


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2408_240855

theorem power_sum_inequality (A B : ℝ) (n : ℕ+) (hA : A ≥ 0) (hB : B ≥ 0) :
  (A + B) ^ (n : ℕ) ≤ 2 ^ (n - 1 : ℕ) * (A ^ (n : ℕ) + B ^ (n : ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2408_240855


namespace NUMINAMATH_CALUDE_nth_term_equation_l2408_240803

theorem nth_term_equation (n : ℕ) : 
  Real.sqrt ((2 * n^2 : ℝ) / (2 * n + 1) - (n - 1)) = Real.sqrt ((n + 1) * (2 * n + 1)) / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_term_equation_l2408_240803


namespace NUMINAMATH_CALUDE_amount_calculation_l2408_240890

theorem amount_calculation (x : ℝ) (amount : ℝ) (h1 : x = 25.0) (h2 : 2 * x = 3 * x - amount) : amount = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l2408_240890


namespace NUMINAMATH_CALUDE_textile_firm_expenses_l2408_240839

/-- Calculates the monthly manufacturing expenses for a textile manufacturing firm. -/
def monthly_manufacturing_expenses (
  num_looms : ℕ
) (total_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease_one_loom : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let cost_saved_one_loom := sales_per_loom - profit_decrease_one_loom
  cost_saved_one_loom * num_looms

/-- Theorem stating the monthly manufacturing expenses for the given problem. -/
theorem textile_firm_expenses :
  monthly_manufacturing_expenses 125 500000 75000 2800 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_textile_firm_expenses_l2408_240839


namespace NUMINAMATH_CALUDE_muffins_per_pack_is_four_l2408_240850

/-- Represents the muffin selling problem --/
structure MuffinProblem where
  total_amount : ℕ -- Total amount to raise in dollars
  muffin_price : ℕ -- Price of each muffin in dollars
  num_cases : ℕ -- Number of cases to sell
  packs_per_case : ℕ -- Number of packs in each case

/-- Calculates the number of muffins in each pack --/
def muffins_per_pack (p : MuffinProblem) : ℕ :=
  (p.total_amount / p.muffin_price) / (p.num_cases * p.packs_per_case)

/-- Theorem stating that the number of muffins per pack is 4 --/
theorem muffins_per_pack_is_four (p : MuffinProblem) 
  (h1 : p.total_amount = 120)
  (h2 : p.muffin_price = 2)
  (h3 : p.num_cases = 5)
  (h4 : p.packs_per_case = 3) : 
  muffins_per_pack p = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_pack_is_four_l2408_240850


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l2408_240884

/-- Theorem about cubic equations and their roots -/
theorem cubic_equation_properties (p q x₀ a b : ℝ) 
  (h1 : x₀^3 + p*x₀ + q = 0)  -- x₀ is a root of the cubic equation
  (h2 : ∀ x, x^3 + p*x + q = (x - x₀)*(x^2 + a*x + b)) :  -- Factorization of the cubic
  (a = x₀) ∧ (p^2 ≥ 4*x₀*q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l2408_240884


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l2408_240814

theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total_students = 36)
  (h2 : chocolate = 12)
  (h3 : apple = 8)
  (h4 : blueberry = 6)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2) : ℚ) / total_students * 360 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l2408_240814


namespace NUMINAMATH_CALUDE_apex_high_debate_points_l2408_240870

theorem apex_high_debate_points :
  ∀ (total_points : ℚ),
  total_points > 0 →
  ∃ (remaining_points : ℕ),
  (1/5 : ℚ) * total_points + (1/3 : ℚ) * total_points + 12 + remaining_points = total_points ∧
  remaining_points ≤ 18 ∧
  remaining_points = 18 :=
by sorry

end NUMINAMATH_CALUDE_apex_high_debate_points_l2408_240870


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l2408_240835

/-- Proves that the speed of a car in the first hour is 60 km/h given the conditions -/
theorem car_speed_first_hour 
  (x : ℝ) -- Speed in the first hour
  (h1 : x > 0) -- Assuming speed is positive
  (h2 : (x + 30) / 2 = 45) -- Average speed equation
  : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l2408_240835


namespace NUMINAMATH_CALUDE_face_card_proportion_l2408_240892

theorem face_card_proportion (p : ℝ) : 
  (p ≥ 0) → (p ≤ 1) → (1 - (1 - p)^3 = 19/27) → p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_face_card_proportion_l2408_240892


namespace NUMINAMATH_CALUDE_parts_per_day_to_finish_ahead_l2408_240893

theorem parts_per_day_to_finish_ahead (total_parts : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_parts_per_day : ℕ) :
  total_parts = 408 →
  total_days = 15 →
  initial_days = 3 →
  initial_parts_per_day = 24 →
  ∃ (x : ℕ), x = 29 ∧ 
    (initial_days * initial_parts_per_day + (total_days - initial_days) * x > total_parts) ∧
    ∀ (y : ℕ), y < x → (initial_days * initial_parts_per_day + (total_days - initial_days) * y ≤ total_parts) :=
by sorry

end NUMINAMATH_CALUDE_parts_per_day_to_finish_ahead_l2408_240893


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2408_240820

theorem min_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 121) : 
  ∃ (a b : ℕ), a^2 - b^2 = 121 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2408_240820


namespace NUMINAMATH_CALUDE_range_of_m_l2408_240888

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2408_240888


namespace NUMINAMATH_CALUDE_inequality_proof_l2408_240869

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2408_240869


namespace NUMINAMATH_CALUDE_residue_of_seven_power_l2408_240857

theorem residue_of_seven_power (n : ℕ) : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_seven_power_l2408_240857


namespace NUMINAMATH_CALUDE_smallest_number_l2408_240847

theorem smallest_number (a b c d : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = -1/3) 
  (hc : c = -2) 
  (hd : d = 0) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2408_240847


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2408_240811

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b/(2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) :=
by sorry

/-- The axis of symmetry of the parabola y = -1/2 x² + x - 5/2 is x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -1/2 * x^2 + x - 5/2
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) ∧ x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2408_240811


namespace NUMINAMATH_CALUDE_inequality_problem_l2408_240848

theorem inequality_problem (x y z : ℝ) (a : ℝ) : 
  (x^2 + y^2 + z^2 = 1) → 
  ((-3 : ℝ) ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧ 
  ((∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) ↔ 
   (a ≥ 4 ∨ a ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2408_240848


namespace NUMINAMATH_CALUDE_money_division_l2408_240866

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * q = 7 * p →
  3 * r = 4 * q →
  q - p = 2800 →
  r - q = 3500 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2408_240866


namespace NUMINAMATH_CALUDE_smallest_expressible_proof_l2408_240822

/-- Represents the number of marbles in each box type -/
def box_sizes : Finset ℕ := {13, 11, 7}

/-- Checks if a number can be expressed as a non-negative integer combination of box sizes -/
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 13 * a + 11 * b + 7 * c

/-- The smallest number such that all larger numbers are expressible -/
def smallest_expressible : ℕ := 30

theorem smallest_expressible_proof :
  (∀ m : ℕ, m > smallest_expressible → is_expressible m) ∧
  (∀ k : ℕ, k < smallest_expressible → ∃ n : ℕ, n > k ∧ ¬is_expressible n) :=
sorry

end NUMINAMATH_CALUDE_smallest_expressible_proof_l2408_240822


namespace NUMINAMATH_CALUDE_post_office_distance_l2408_240819

/-- Proves that the distance of a round trip is 20 km given specific speeds and total time -/
theorem post_office_distance (outbound_speed inbound_speed : ℝ) (total_time : ℝ) 
  (h1 : outbound_speed = 25)
  (h2 : inbound_speed = 4)
  (h3 : total_time = 5.8) :
  let distance := (outbound_speed * inbound_speed * total_time) / (outbound_speed + inbound_speed)
  distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_post_office_distance_l2408_240819


namespace NUMINAMATH_CALUDE_proposition_one_is_correct_l2408_240846

theorem proposition_one_is_correct (p q : Prop) :
  (¬(p ∧ q) ∧ ¬(p ∨ q)) → (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_one_is_correct_l2408_240846


namespace NUMINAMATH_CALUDE_seokjin_drank_least_l2408_240805

def seokjin_milk : ℚ := 11/10
def jungkook_milk : ℚ := 13/10
def yoongi_milk : ℚ := 7/6

theorem seokjin_drank_least :
  seokjin_milk < jungkook_milk ∧ seokjin_milk < yoongi_milk :=
by sorry

end NUMINAMATH_CALUDE_seokjin_drank_least_l2408_240805


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l2408_240831

-- Define the set of points satisfying the given inequalities
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≥ 3 * p.1 ∧ p.2 ≥ 5 - p.1 ∧ p.2 < 7}

-- Define Quadrant I
def QuadrantI : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrant_I : S ⊆ QuadrantI := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l2408_240831


namespace NUMINAMATH_CALUDE_not_prime_polynomial_l2408_240854

theorem not_prime_polynomial (x y : ℤ) : 
  ¬ (Nat.Prime (x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_polynomial_l2408_240854


namespace NUMINAMATH_CALUDE_correct_list_price_l2408_240845

/-- The list price of the item -/
def list_price : ℝ := 45

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.1

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.15

/-- Theorem stating that the list price is correct -/
theorem correct_list_price :
  alice_rate * alice_price list_price = bob_rate * bob_price list_price :=
by sorry

end NUMINAMATH_CALUDE_correct_list_price_l2408_240845


namespace NUMINAMATH_CALUDE_optimal_price_and_quantity_l2408_240872

/-- Represents the sales and pricing model for a product -/
structure SalesModel where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialSalesVolume : ℝ
  priceElasticity : ℝ
  targetProfit : ℝ
  maxCost : ℝ

/-- Calculates the sales volume for a given selling price -/
def salesVolume (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialSalesVolume - model.priceElasticity * (sellingPrice - model.initialSellingPrice)

/-- Calculates the profit for a given selling price -/
def profit (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - model.initialPurchasePrice) * (salesVolume model sellingPrice)

/-- Calculates the cost for a given selling price -/
def cost (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialPurchasePrice * (salesVolume model sellingPrice)

/-- Theorem stating that the optimal selling price and purchase quantity satisfy the constraints -/
theorem optimal_price_and_quantity (model : SalesModel) 
  (h_model : model = { 
    initialPurchasePrice := 40,
    initialSellingPrice := 50,
    initialSalesVolume := 500,
    priceElasticity := 10,
    targetProfit := 8000,
    maxCost := 10000
  }) :
  ∃ (optimalPrice optimalQuantity : ℝ),
    optimalPrice = 80 ∧
    optimalQuantity = 200 ∧
    profit model optimalPrice = model.targetProfit ∧
    cost model optimalPrice ≤ model.maxCost :=
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_quantity_l2408_240872


namespace NUMINAMATH_CALUDE_candies_remaining_is_155_l2408_240865

/-- The number of candies remaining after Carlos ate his share -/
def candies_remaining : ℕ :=
  let red : ℕ := 60
  let yellow : ℕ := 3 * red - 30
  let blue : ℕ := (2 * yellow) / 4
  let green : ℕ := 40
  let purple : ℕ := green / 3
  let silver : ℕ := 15
  let gold : ℕ := silver / 2
  let total : ℕ := red + yellow + blue + green + purple + silver + gold
  let eaten : ℕ := yellow + (green * 3 / 4) + (blue / 3)
  total - eaten

theorem candies_remaining_is_155 : candies_remaining = 155 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_is_155_l2408_240865


namespace NUMINAMATH_CALUDE_race_orders_theorem_l2408_240833

-- Define the number of racers
def num_racers : ℕ := 6

-- Define the function to calculate the number of possible orders
def possible_orders (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_orders_theorem : possible_orders num_racers = 720 := by
  sorry

end NUMINAMATH_CALUDE_race_orders_theorem_l2408_240833


namespace NUMINAMATH_CALUDE_percentage_relation_l2408_240823

theorem percentage_relation (x y : ℕ+) (h1 : y * x = 100 * 100) (h2 : y = 125) :
  (y : ℝ) / ((25 : ℝ) / 100 * x) * 100 = 625 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2408_240823


namespace NUMINAMATH_CALUDE_max_shelves_with_five_books_together_l2408_240879

/-- Given 1300 books and k shelves, this theorem states that 18 is the largest value of k
    for which there will always be at least 5 books on the same shelf
    before and after any rearrangement. -/
theorem max_shelves_with_five_books_together (k : ℕ) : 
  (∀ (arrangement₁ arrangement₂ : Fin k → Fin 1300 → Prop), 
    (∀ b, ∃! s, arrangement₁ s b) → 
    (∀ b, ∃! s, arrangement₂ s b) → 
    (∃ s : Fin k, ∃ (books : Finset (Fin 1300)), 
      books.card = 5 ∧ 
      (∀ b ∈ books, arrangement₁ s b ∧ arrangement₂ s b))) ↔ 
  k ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_shelves_with_five_books_together_l2408_240879


namespace NUMINAMATH_CALUDE_total_problems_eq_480_l2408_240873

/-- The number of math problems Marvin solved yesterday -/
def marvin_yesterday : ℕ := 40

/-- The number of math problems Marvin solved today -/
def marvin_today : ℕ := 3 * marvin_yesterday

/-- The total number of math problems Marvin solved over two days -/
def marvin_total : ℕ := marvin_yesterday + marvin_today

/-- The number of math problems Arvin solved over two days -/
def arvin_total : ℕ := 2 * marvin_total

/-- The total number of math problems solved by both Marvin and Arvin -/
def total_problems : ℕ := marvin_total + arvin_total

theorem total_problems_eq_480 : total_problems = 480 := by sorry

end NUMINAMATH_CALUDE_total_problems_eq_480_l2408_240873


namespace NUMINAMATH_CALUDE_distribute_five_items_three_bags_l2408_240859

/-- The number of ways to distribute n distinct items into k identical bags --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 distinct items into 3 identical bags results in 51 ways --/
theorem distribute_five_items_three_bags : distribute 5 3 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_three_bags_l2408_240859


namespace NUMINAMATH_CALUDE_cards_lost_l2408_240826

theorem cards_lost (initial_cards : ℝ) (final_cards : ℕ) : 
  initial_cards = 47.0 → final_cards = 40 → initial_cards - final_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l2408_240826


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l2408_240804

theorem four_digit_number_problem (n : ℕ) : 
  (1000 ≤ n) ∧ (n < 10000) ∧  -- n is a four-digit number
  (n % 10 = 9) ∧              -- the ones digit of n is 9
  ((n - 3) + 57 = 1823)       -- the sum of the mistaken number and 57 is 1823
  → n = 1769 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l2408_240804


namespace NUMINAMATH_CALUDE_max_viewers_per_week_l2408_240841

/-- Represents the number of times a series is broadcast per week -/
structure BroadcastCount where
  seriesA : ℕ
  seriesB : ℕ

/-- Calculates the total program time for a given broadcast count -/
def totalProgramTime (bc : BroadcastCount) : ℕ :=
  80 * bc.seriesA + 40 * bc.seriesB

/-- Calculates the total commercial time for a given broadcast count -/
def totalCommercialTime (bc : BroadcastCount) : ℕ :=
  bc.seriesA + bc.seriesB

/-- Calculates the total number of viewers for a given broadcast count -/
def totalViewers (bc : BroadcastCount) : ℕ :=
  600000 * bc.seriesA + 200000 * bc.seriesB

/-- Represents the constraints for the broadcast schedule -/
def validBroadcastCount (bc : BroadcastCount) : Prop :=
  totalProgramTime bc ≤ 320 ∧ totalCommercialTime bc ≥ 6

/-- Theorem: The maximum number of viewers per week is 2,000,000 -/
theorem max_viewers_per_week :
  ∃ (bc : BroadcastCount), validBroadcastCount bc ∧
  ∀ (bc' : BroadcastCount), validBroadcastCount bc' →
  totalViewers bc' ≤ 2000000 :=
sorry

end NUMINAMATH_CALUDE_max_viewers_per_week_l2408_240841


namespace NUMINAMATH_CALUDE_min_value_sequence_l2408_240858

theorem min_value_sequence (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∀ k, ∃ r, a (k + 1) = a k + r) →  -- Arithmetic progression
  (∀ k, ∃ q, a (k + 1) = a k * q) →  -- Geometric progression
  (a 7 = a 6 + 2 * a 5) →  -- Given condition
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Given condition
  (∃ min_val : ℝ, min_val = 1 + Real.sqrt 5 / 3 ∧
    ∀ p q : ℕ, 1 / p + 5 / q ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sequence_l2408_240858


namespace NUMINAMATH_CALUDE_complex_sum_power_l2408_240849

theorem complex_sum_power (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_power_l2408_240849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2408_240895

/-- 
Given an arithmetic sequence with:
- 20 terms
- First term is 4
- Sum of the sequence is 650

Prove that the common difference is 3
-/
theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ),
  (20 : ℚ) / 2 * (2 * 4 + (20 - 1) * d) = 650 →
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2408_240895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2408_240868

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence with common ratio q
  abs q > 1 →                   -- |q| > 1
  a 2 + a 7 = 2 →               -- a₂ + a₇ = 2
  a 4 * a 5 = -15 →             -- a₄a₅ = -15
  a 12 = -25 / 3 :=              -- a₁₂ = -25/3
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2408_240868


namespace NUMINAMATH_CALUDE_math_teacher_initial_amount_l2408_240806

theorem math_teacher_initial_amount :
  let basic_calculator_cost : ℕ := 8
  let scientific_calculator_cost : ℕ := 2 * basic_calculator_cost
  let graphing_calculator_cost : ℕ := 3 * scientific_calculator_cost
  let total_cost : ℕ := basic_calculator_cost + scientific_calculator_cost + graphing_calculator_cost
  let change : ℕ := 28
  let initial_amount : ℕ := total_cost + change
  initial_amount = 100
  := by sorry

end NUMINAMATH_CALUDE_math_teacher_initial_amount_l2408_240806


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2408_240810

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation 2x + y - 2 = 0
    passes through the point (1,0) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y - 2 = 0) →  -- Equation of line L1
  (2*x + y - 2 = 0) →  -- Equation of line L2
  (2*1 + 0 - 2 = 0) ∧  -- L2 passes through (1,0)
  (1 * 2 = -1) -- Slopes of L1 and L2 are negative reciprocals
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2408_240810


namespace NUMINAMATH_CALUDE_die_product_divisibility_l2408_240863

theorem die_product_divisibility : 
  ∀ (S : Finset ℕ), 
  S ⊆ Finset.range 9 → 
  S.card = 7 → 
  48 ∣ S.prod id := by
sorry

end NUMINAMATH_CALUDE_die_product_divisibility_l2408_240863


namespace NUMINAMATH_CALUDE_line_equation_proof_l2408_240828

theorem line_equation_proof (x y : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  let angle : ℝ := π / 4  -- 45° in radians
  let slope : ℝ := Real.tan angle
  (x - y + 3 = 0) ↔ 
    (y - P.2 = slope * (x - P.1) ∧ slope = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2408_240828


namespace NUMINAMATH_CALUDE_multiplication_result_l2408_240821

theorem multiplication_result : 163861 * 454733 = 74505853393 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l2408_240821


namespace NUMINAMATH_CALUDE_reciprocal_squares_sum_of_product_five_l2408_240852

theorem reciprocal_squares_sum_of_product_five (a b : ℕ) (h : a * b = 5) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 26 / 25 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_squares_sum_of_product_five_l2408_240852


namespace NUMINAMATH_CALUDE_sum_of_digits_of_9n_l2408_240842

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n / 10^i) % 10 < (n / 10^j) % 10

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem stating that for any natural number with strictly increasing digits,
    the sum of digits of 9 times that number is always 9 -/
theorem sum_of_digits_of_9n (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_9n_l2408_240842


namespace NUMINAMATH_CALUDE_no_perfect_square_pairs_l2408_240880

theorem no_perfect_square_pairs : ¬∃ (x y : ℕ+), ∃ (z : ℕ+), (x * y + 1) * (x * y + x + 2) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_pairs_l2408_240880


namespace NUMINAMATH_CALUDE_lego_pieces_sold_l2408_240886

/-- The number of single Lego pieces sold -/
def single_pieces : ℕ := sorry

/-- The total earnings in cents -/
def total_earnings : ℕ := 1000

/-- The number of double pieces sold -/
def double_pieces : ℕ := 45

/-- The number of triple pieces sold -/
def triple_pieces : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_pieces : ℕ := 165

/-- The cost of each circle in cents -/
def circle_cost : ℕ := 1

theorem lego_pieces_sold :
  single_pieces = 100 :=
by sorry

end NUMINAMATH_CALUDE_lego_pieces_sold_l2408_240886


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_eq_a_pow_three_fourths_l2408_240871

theorem sqrt_a_sqrt_a_eq_a_pow_three_fourths (a : ℝ) (h : a > 0) :
  Real.sqrt (a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_eq_a_pow_three_fourths_l2408_240871


namespace NUMINAMATH_CALUDE_find_constant_b_l2408_240844

theorem find_constant_b (a b c : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 2)*(a*x^2 + b*x + c) = 9*x^4 - 10*x^3 + 5*x^2 - 8*x + 4) → 
  b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_b_l2408_240844


namespace NUMINAMATH_CALUDE_max_non_zero_numbers_eq_sum_binary_digits_l2408_240812

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- The game state -/
structure GameState where
  numbers : List ℕ

/-- The game move -/
inductive Move
  | Sum : ℕ → ℕ → Move
  | Diff : ℕ → ℕ → Move

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Check if the game is over -/
def isGameOver (state : GameState) : Bool := sorry

/-- The maximum number of non-zero numbers at the end of the game -/
def maxNonZeroNumbers (initialOnes : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_non_zero_numbers_eq_sum_binary_digits :
  maxNonZeroNumbers 2020 = sumBinaryDigits 2020 := by sorry

end NUMINAMATH_CALUDE_max_non_zero_numbers_eq_sum_binary_digits_l2408_240812


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2408_240899

theorem quadratic_minimum (h : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) →
  h = -2 ∨ h = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2408_240899


namespace NUMINAMATH_CALUDE_ellipse_sum_l2408_240877

/-- The sum of h, k, a, and b for a specific ellipse -/
theorem ellipse_sum (h k a b : ℝ) : 
  ((3 : ℝ) = h) → ((-5 : ℝ) = k) → ((7 : ℝ) = a) → ((2 : ℝ) = b) → 
  h + k + a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2408_240877


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2408_240853

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧
    radius = Real.sqrt 5 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2408_240853


namespace NUMINAMATH_CALUDE_special_rectangle_area_l2408_240827

/-- Rectangle with a special circle configuration -/
structure SpecialRectangle where
  -- The radius of the inscribed circle
  r : ℝ
  -- The width of the rectangle (length of side AB)
  w : ℝ
  -- The height of the rectangle (length of side AD)
  h : ℝ
  -- The circle is tangent to sides AD and BC
  tangent_sides : h = 2 * r
  -- The circle is tangent internally to the semicircle with diameter AB
  tangent_semicircle : w = 6 * r
  -- The circle passes through the midpoint of AB
  passes_midpoint : w / 2 = 3 * r

/-- The area of the special rectangle is 12r^2 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = 12 * rect.r^2 := by
  sorry


end NUMINAMATH_CALUDE_special_rectangle_area_l2408_240827


namespace NUMINAMATH_CALUDE_max_sum_of_squared_unit_complex_l2408_240897

theorem max_sum_of_squared_unit_complex (z : ℂ) (a b : ℝ) 
  (h1 : Complex.abs z = 1)
  (h2 : z^2 = Complex.mk a b) :
  ∃ (x y : ℝ), Complex.mk x y = z^2 ∧ x + y ≤ Real.sqrt 2 ∧
  ∀ (c d : ℝ), Complex.mk c d = z^2 → c + d ≤ x + y :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_unit_complex_l2408_240897


namespace NUMINAMATH_CALUDE_largest_fraction_l2408_240836

theorem largest_fraction : 
  let fractions := [1/5, 2/10, 7/15, 9/20, 3/6]
  ∀ x ∈ fractions, x ≤ (3:ℚ)/6 := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l2408_240836


namespace NUMINAMATH_CALUDE_ab_length_l2408_240815

-- Define the triangles
structure Triangle :=
  (a b c : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles ABC and DEF
def ABC : Triangle := { a := 7, b := 14, c := 10 }
def DEF : Triangle := { a := 6, b := 3, c := 5 }

-- State the theorem
theorem ab_length :
  similar ABC DEF →
  ABC.b = 14 →
  DEF.a = 6 →
  DEF.b = 3 →
  ABC.a = 7 := by sorry

end NUMINAMATH_CALUDE_ab_length_l2408_240815


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2408_240898

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2408_240898


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2408_240861

theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n+1) + X^(n+2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2408_240861


namespace NUMINAMATH_CALUDE_inequality_proof_l2408_240891

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : 1 / a + 1 / b = 1) (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2408_240891


namespace NUMINAMATH_CALUDE_min_value_theorem_l2408_240813

theorem min_value_theorem (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (b + c) / a + (a + c) / b = (a + b) / c + 1) :
  (∀ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ (y + z) / x + (x + z) / y = (x + y) / z + 1 → (a + b) / c ≤ (x + y) / z) ∧
  (a + b) / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2408_240813


namespace NUMINAMATH_CALUDE_midpoint_cut_equal_parts_l2408_240856

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  h_positive : 0 < length ∧ 0 < width
  h_length_gt_width : length > width

/-- Represents the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents a cut parallel to the shorter side of the rectangle -/
def parallel_cut (r : Rectangle) (x : ℝ) : ℝ := x * r.width

/-- Theorem stating that cutting a rectangle at its midpoint results in two equal parts -/
theorem midpoint_cut_equal_parts (r : Rectangle) :
  parallel_cut r (r.length / 2) = r.area / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_cut_equal_parts_l2408_240856


namespace NUMINAMATH_CALUDE_hundredths_place_of_seven_twentieths_l2408_240882

theorem hundredths_place_of_seven_twentieths (x : ℚ) : 
  x = 7 / 20 → (x * 100).floor % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hundredths_place_of_seven_twentieths_l2408_240882


namespace NUMINAMATH_CALUDE_cos_seven_pi_four_l2408_240881

theorem cos_seven_pi_four : Real.cos (7 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_four_l2408_240881


namespace NUMINAMATH_CALUDE_subset_necessary_not_sufficient_l2408_240896

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem subset_necessary_not_sufficient (a : ℕ) :
  (A a ⊆ B ↔ a = 3) ↔ False ∧
  (a = 3 → A a ⊆ B) ∧
  ¬(A a ⊆ B → a = 3) :=
sorry

end NUMINAMATH_CALUDE_subset_necessary_not_sufficient_l2408_240896


namespace NUMINAMATH_CALUDE_tournament_games_32_teams_l2408_240840

/-- The number of games needed in a single-elimination tournament to declare a winner -/
def games_needed (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are needed to declare a winner -/
theorem tournament_games_32_teams :
  games_needed 32 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_32_teams_l2408_240840


namespace NUMINAMATH_CALUDE_second_class_average_l2408_240800

/-- Given two classes of students, this theorem proves the average mark of the second class. -/
theorem second_class_average (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (combined_avg : ℝ) 
  (h1 : students1 = 58)
  (h2 : students2 = 52)
  (h3 : avg1 = 67)
  (h4 : combined_avg = 74.0909090909091) : 
  ∃ (avg2 : ℝ), abs (avg2 - 81.62) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_class_average_l2408_240800


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l2408_240825

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_of_geometric_terms
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum : a 3 + a 11 = 8)
  (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l2408_240825


namespace NUMINAMATH_CALUDE_xyz_value_l2408_240818

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) :
  x * y * z = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2408_240818


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2408_240808

theorem quadratic_form_k_value (x : ℝ) : 
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2408_240808


namespace NUMINAMATH_CALUDE_plot_area_approx_360_l2408_240834

/-- Calculates the area of a rectangular plot given its breadth, where the length is 25% less than the breadth -/
def plot_area (breadth : ℝ) : ℝ :=
  let length := 0.75 * breadth
  length * breadth

/-- The breadth of the plot -/
def plot_breadth : ℝ := 21.908902300206645

/-- Theorem stating that the area of the plot is approximately 360 square meters -/
theorem plot_area_approx_360 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |plot_area plot_breadth - 360| < ε :=
sorry

end NUMINAMATH_CALUDE_plot_area_approx_360_l2408_240834


namespace NUMINAMATH_CALUDE_total_distance_run_l2408_240807

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 50

/-- The number of pairs of children (boy-girl pairs) -/
def num_pairs : ℕ := 4

/-- Theorem: The total distance run by all children is 100 meters -/
theorem total_distance_run : 
  track_length * (num_pairs : ℝ) / 2 = 100 := by sorry

end NUMINAMATH_CALUDE_total_distance_run_l2408_240807


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l2408_240887

/-- A quadratic function f(x) = -x² + 2x + c --/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- The y-coordinate of a point (x, f(x)) on the graph of f --/
def y (c : ℝ) (x : ℝ) : ℝ := f c x

theorem quadratic_points_relationship (c : ℝ) :
  let y₁ := y c (-1)
  let y₂ := y c 3
  let y₃ := y c 5
  y₁ = y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l2408_240887


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l2408_240883

/-- The sum of the lengths of quarter-circles approaches πC as n approaches infinity -/
theorem quarter_circles_sum_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * (C / (2 * π * n)) / 4) - π * C| < ε :=
by sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l2408_240883


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2408_240816

theorem unfair_coin_probability (n : ℕ) (k : ℕ) (p_head : ℚ) (p_tail : ℚ) :
  n = 8 →
  k = 3 →
  p_head = 1/3 →
  p_tail = 2/3 →
  p_head + p_tail = 1 →
  (n.choose k : ℚ) * p_tail^k * p_head^(n-k) = 448/177147 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l2408_240816


namespace NUMINAMATH_CALUDE_exists_four_digit_number_sum_12_div_5_l2408_240862

/-- A four-digit number is represented as a tuple of four natural numbers -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Check if a given four-digit number has digits that add up to 12 -/
def digits_sum_to_12 (n : FourDigitNumber) : Prop :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2 = 12

/-- Check if a given four-digit number is divisible by 5 -/
def divisible_by_5 (n : FourDigitNumber) : Prop :=
  (n.1 * 1000 + n.2.1 * 100 + n.2.2.1 * 10 + n.2.2.2) % 5 = 0

/-- Check if a given number is a valid four-digit number (between 1000 and 9999) -/
def is_valid_four_digit (n : FourDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.1 ≤ 9 ∧ n.2.1 ≤ 9 ∧ n.2.2.1 ≤ 9 ∧ n.2.2.2 ≤ 9

theorem exists_four_digit_number_sum_12_div_5 :
  ∃ (n : FourDigitNumber), is_valid_four_digit n ∧ digits_sum_to_12 n ∧ divisible_by_5 n :=
by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_number_sum_12_div_5_l2408_240862
