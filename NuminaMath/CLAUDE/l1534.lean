import Mathlib

namespace complex_number_equality_l1534_153463

theorem complex_number_equality : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ z = -1 + Complex.I := by
  sorry

end complex_number_equality_l1534_153463


namespace row_swap_matrix_exists_l1534_153472

open Matrix

theorem row_swap_matrix_exists : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
  N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end row_swap_matrix_exists_l1534_153472


namespace factorial_500_trailing_zeroes_l1534_153437

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l1534_153437


namespace sum_of_roots_cubic_polynomial_l1534_153478

theorem sum_of_roots_cubic_polynomial : 
  let p (x : ℝ) := 3 * x^3 - 9 * x^2 - 72 * x - 18
  ∃ (r s t : ℝ), p r = 0 ∧ p s = 0 ∧ p t = 0 ∧ r + s + t = 3 :=
by sorry

end sum_of_roots_cubic_polynomial_l1534_153478


namespace exponential_decreasing_l1534_153498

theorem exponential_decreasing (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end exponential_decreasing_l1534_153498


namespace gear_alignment_theorem_l1534_153428

/-- Represents a gear with a certain number of teeth and ground-off pairs -/
structure Gear where
  initial_teeth : Nat
  ground_off_pairs : Nat

/-- Calculates the number of remaining teeth on a gear -/
def remaining_teeth (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs

/-- Calculates the number of possible alignment positions -/
def alignment_positions (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs + 1

/-- Theorem stating that there exists exactly one position where a hole in one gear
    aligns with a whole tooth on the other gear -/
theorem gear_alignment_theorem (g1 g2 : Gear)
  (h1 : g1.initial_teeth = 32)
  (h2 : g2.initial_teeth = 32)
  (h3 : g1.ground_off_pairs = 6)
  (h4 : g2.ground_off_pairs = 6)
  : ∃! position, position ≤ alignment_positions g1 ∧
    (position ≠ 0 → 
      (∃ hole_in_g1 whole_tooth_in_g2, 
        hole_in_g1 ≤ g1.ground_off_pairs ∧
        whole_tooth_in_g2 ≤ remaining_teeth g2 ∧
        hole_in_g1 ≠ whole_tooth_in_g2)) :=
  sorry

end gear_alignment_theorem_l1534_153428


namespace diana_work_hours_l1534_153418

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday combined
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday combined
  weekly_earnings : ℕ    -- Weekly earnings in dollars
  hourly_rate : ℕ        -- Hourly rate in dollars

/-- Theorem stating Diana's work hours on Monday, Wednesday, and Friday --/
theorem diana_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 30)  -- 15 hours each on Tuesday and Thursday
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30)
  : schedule.mon_wed_fri_hours = 30 := by
  sorry


end diana_work_hours_l1534_153418


namespace probability_allison_wins_l1534_153485

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ
| 0 => 3
| 1 => 3
| _ => 5

def prob_brian_less_or_equal_5 : ℚ := 5 / 6

def prob_noah_less_or_equal_5 : ℚ := 1

theorem probability_allison_wins : ℚ := by
  sorry

end probability_allison_wins_l1534_153485


namespace fraction_sum_equals_negative_one_l1534_153492

theorem fraction_sum_equals_negative_one (a : ℝ) (h : 1 - 2*a ≠ 0) :
  a / (1 - 2*a) + (a - 1) / (1 - 2*a) = -1 := by
  sorry

end fraction_sum_equals_negative_one_l1534_153492


namespace evaluate_expression_l1534_153491

theorem evaluate_expression : (3025^2 : ℝ) / (305^2 - 295^2) = 1525.10417 := by
  sorry

end evaluate_expression_l1534_153491


namespace sum_of_fractions_equals_seven_l1534_153411

theorem sum_of_fractions_equals_seven :
  let S := 1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 
           1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)
  S = 7 := by
sorry

end sum_of_fractions_equals_seven_l1534_153411


namespace hannah_seashell_distribution_l1534_153445

theorem hannah_seashell_distribution (noah liam hannah : ℕ) : 
  hannah = 4 * liam ∧ 
  liam = 3 * noah → 
  (7 : ℚ) / 36 = (hannah + liam + noah) / 3 - liam / hannah :=
by sorry

end hannah_seashell_distribution_l1534_153445


namespace min_distance_line_parabola_l1534_153446

/-- The minimum distance between a point on the line x - y - 4 = 0 and a point on the parabola x² = 4y is (3√2)/2 -/
theorem min_distance_line_parabola :
  let line := {p : ℝ × ℝ | p.1 - p.2 = 4}
  let parabola := {p : ℝ × ℝ | p.1^2 = 4 * p.2}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ parabola ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ parabola →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end min_distance_line_parabola_l1534_153446


namespace sum_of_odd_terms_l1534_153416

theorem sum_of_odd_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n, S n = n^2 + n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 50 :=
by sorry

end sum_of_odd_terms_l1534_153416


namespace set_M_characterization_l1534_153427

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 ∨ y = 1}

-- Define the set of valid x values
def valid_x : Set ℝ := {x | x ≠ 1 ∧ x ≠ -1}

-- Theorem statement
theorem set_M_characterization : 
  ∀ x : ℝ, (x^2 ∈ M ∧ 1 ∈ M ∧ x^2 ≠ 1) ↔ x ∈ valid_x :=
sorry

end set_M_characterization_l1534_153427


namespace bouquets_calculation_l1534_153488

/-- Given the initial number of flowers, flowers per bouquet, and wilted flowers,
    calculates the number of bouquets that can be made. -/
def calculate_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Proves that given 88 initial flowers, 5 flowers per bouquet, and 48 wilted flowers,
    the number of bouquets that can be made is equal to 8. -/
theorem bouquets_calculation :
  calculate_bouquets 88 5 48 = 8 := by
  sorry

end bouquets_calculation_l1534_153488


namespace equation_solutions_l1534_153471

theorem equation_solutions :
  (∀ x, x * (x - 5) = 3 * x - 15 ↔ x = 5 ∨ x = 3) ∧
  (∀ y, 2 * y^2 - 9 * y + 5 = 0 ↔ y = (9 + Real.sqrt 41) / 4 ∨ y = (9 - Real.sqrt 41) / 4) :=
by sorry

end equation_solutions_l1534_153471


namespace ribbon_length_l1534_153473

theorem ribbon_length (A : ℝ) (π : ℝ) (h1 : A = 616) (h2 : π = 22 / 7) : 
  let r := Real.sqrt (A / π)
  let C := 2 * π * r
  C + 5 = 93 := by sorry

end ribbon_length_l1534_153473


namespace complex_subtraction_simplification_l1534_153448

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end complex_subtraction_simplification_l1534_153448


namespace twin_birthday_product_l1534_153457

theorem twin_birthday_product (age : ℕ) (h : age = 5) :
  (age + 1) * (age + 1) - age * age = 11 := by
  sorry

end twin_birthday_product_l1534_153457


namespace f_properties_l1534_153466

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  (∀ k : ℤ, ∀ x : ℝ, -π/3 + k * π ≤ x ∧ x ≤ π/6 + k * π → 
    ∀ y : ℝ, -π/3 + k * π ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ A B C a b c : ℝ, 
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
    (a + 2*c) * Real.cos B = -b * Real.cos A →
    2 < f A ∧ f A ≤ 3) :=
sorry

end f_properties_l1534_153466


namespace car_travel_distance_l1534_153443

/-- Proves that two cars traveling at different speeds for different times cover the same distance of 600 miles -/
theorem car_travel_distance :
  ∀ (distance : ℝ) (time_R : ℝ) (speed_R : ℝ),
    speed_R = 50 →
    distance = speed_R * time_R →
    distance = (speed_R + 10) * (time_R - 2) →
    distance = 600 := by
  sorry

end car_travel_distance_l1534_153443


namespace mets_to_red_sox_ratio_l1534_153438

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of Mets to Red Sox fans -/
theorem mets_to_red_sox_ratio (fans : FanCount) 
  (total_fans : fans.yankees + fans.mets + fans.red_sox = 330)
  (yankees_to_mets : Ratio)
  (yankees_mets_ratio : yankees_to_mets.numerator * fans.mets = yankees_to_mets.denominator * fans.yankees)
  (yankees_mets_values : yankees_to_mets.numerator = 3 ∧ yankees_to_mets.denominator = 2)
  (mets_count : fans.mets = 88) :
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧ 
    r.numerator * fans.red_sox = r.denominator * fans.mets :=
sorry

end mets_to_red_sox_ratio_l1534_153438


namespace exists_min_value_l1534_153490

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

/-- Theorem stating that there exists a minimum value for the function -/
theorem exists_min_value :
  ∃ (y : ℝ), ∃ (min_val : ℝ), ∀ (x : ℝ), f x y ≥ min_val :=
sorry

end exists_min_value_l1534_153490


namespace complex_arithmetic_expression_l1534_153451

theorem complex_arithmetic_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 := by
  sorry

end complex_arithmetic_expression_l1534_153451


namespace vikki_take_home_pay_l1534_153444

def vikki_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := tax_rate * gross_earnings
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  let take_home_pay := gross_earnings - total_deductions
  take_home_pay = 310

theorem vikki_take_home_pay :
  vikki_problem 42 10 (20/100) (5/100) 5 :=
sorry

end vikki_take_home_pay_l1534_153444


namespace quadratic_equation_solutions_l1534_153476

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end quadratic_equation_solutions_l1534_153476


namespace optimal_price_reduction_l1534_153405

/-- Represents the price reduction of thermal shirts -/
def price_reduction : ℝ := 20

/-- Initial average daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per dollar of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target daily profit -/
def target_profit : ℝ := 1200

/-- New daily sales after price reduction -/
def new_sales (x : ℝ) : ℝ := initial_sales + sales_increase_rate * x

/-- New profit per piece after price reduction -/
def new_profit_per_piece (x : ℝ) : ℝ := initial_profit_per_piece - x

/-- Daily profit function -/
def daily_profit (x : ℝ) : ℝ := new_sales x * new_profit_per_piece x

theorem optimal_price_reduction :
  daily_profit price_reduction = target_profit ∧
  ∀ y, y ≠ price_reduction → daily_profit y ≤ daily_profit price_reduction :=
by sorry

end optimal_price_reduction_l1534_153405


namespace optimal_selling_price_l1534_153475

-- Define the initial conditions
def initial_purchase_price : ℝ := 10
def initial_selling_price : ℝ := 18
def initial_daily_sales : ℝ := 60

-- Define the price-sales relationships
def price_increase_effect (price_change : ℝ) : ℝ := -5 * price_change
def price_decrease_effect (price_change : ℝ) : ℝ := 10 * price_change

-- Define the profit functions
def profit_function_high (x : ℝ) : ℝ := -5 * (x - 20)^2 + 500
def profit_function_low (x : ℝ) : ℝ := -10 * (x - 17)^2 + 490

-- Theorem statement
theorem optimal_selling_price :
  ∃ (x : ℝ), x = 20 ∧
  ∀ (y : ℝ), y ≥ initial_selling_price →
    profit_function_high y ≤ profit_function_high x ∧
  ∀ (z : ℝ), z < initial_selling_price →
    profit_function_low z ≤ profit_function_high x :=
by sorry

end optimal_selling_price_l1534_153475


namespace quadratic_expression_a_range_l1534_153455

-- Define the quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for the inequality solution
def inequality_solution (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > -2 * x ↔ 1 < x ∧ x < 3

-- Theorem 1
theorem quadratic_expression
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ x, quadratic_function a b c x + 6 * a = 0 ∧
              ∀ y, quadratic_function a b c y + 6 * a = 0 → y = x) :
  ∃ x, quadratic_function (-1/5) (-6/5) (-3/5) x = quadratic_function a b c x :=
sorry

-- Theorem 2
theorem a_range
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ m, ∀ x, quadratic_function a b c x ≤ m ∧ m > 0) :
  a > -2 + Real.sqrt 3 ∨ a < -2 - Real.sqrt 3 :=
sorry

end quadratic_expression_a_range_l1534_153455


namespace monotonicity_condition_max_value_condition_l1534_153408

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^3 + 9 * x

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ m ≥ 3 :=
sorry

-- Part 2: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x ∈ Set.Icc 1 2, f m x ≤ 4) ∧
           (∃ x ∈ Set.Icc 1 2, f m x = 4) ∧
           m = -2 :=
sorry

end monotonicity_condition_max_value_condition_l1534_153408


namespace knitting_productivity_difference_l1534_153458

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of cycles in a given time period -/
def cyclesInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  period / cycleTime k

/-- Calculates the total working time in a given period -/
def workingTimeInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  k.workTime * cyclesInPeriod k period

/-- Theorem stating the productivity difference between two knitters -/
theorem knitting_productivity_difference
  (girl1 : Knitter)
  (girl2 : Knitter)
  (h1 : girl1.workTime = 5)
  (h2 : girl2.workTime = 7)
  (h3 : girl1.breakTime = 1)
  (h4 : girl2.breakTime = 1)
  (h5 : ∃ (t : ℕ), workingTimeInPeriod girl1 t = workingTimeInPeriod girl2 t) :
  (workingTimeInPeriod girl2 24 : ℚ) / (workingTimeInPeriod girl1 24 : ℚ) = 21 / 20 := by
  sorry

end knitting_productivity_difference_l1534_153458


namespace telescope_visual_range_l1534_153480

/-- Given a telescope that increases the visual range by 150 percent from an original range of 60 kilometers, 
    the new visual range is 150 kilometers. -/
theorem telescope_visual_range : 
  let original_range : ℝ := 60
  let increase_percent : ℝ := 150
  let new_range : ℝ := original_range * (1 + increase_percent / 100)
  new_range = 150 := by sorry

end telescope_visual_range_l1534_153480


namespace ellipse_foci_range_l1534_153439

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where the foci are on the x-axis,
    prove that the range of m is (-3, 0) ∪ (0, 3) -/
theorem ellipse_foci_range (m : ℝ) : 
  (∃ x y : ℝ, x^2/9 + y^2/m^2 = 1 ∧ (∃ c : ℝ, c > 0 ∧ c < 3 ∧ 
    (∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → x^2 - y^2 = c^2))) ↔ 
  m ∈ Set.union (Set.Ioo (-3) 0) (Set.Ioo 0 3) :=
sorry

end ellipse_foci_range_l1534_153439


namespace xiaotong_grade_l1534_153489

/-- Represents the grading system for a physical education course -/
structure GradingSystem where
  maxScore : ℝ
  classroomWeight : ℝ
  midtermWeight : ℝ
  finalWeight : ℝ

/-- Represents a student's scores in the physical education course -/
structure StudentScores where
  classroom : ℝ
  midterm : ℝ
  final : ℝ

/-- Calculates the final grade based on the grading system and student scores -/
def calculateGrade (sys : GradingSystem) (scores : StudentScores) : ℝ :=
  sys.classroomWeight * scores.classroom +
  sys.midtermWeight * scores.midterm +
  sys.finalWeight * scores.final

/-- The theorem stating that Xiaotong's grade is 55 given the specified grading system and scores -/
theorem xiaotong_grade :
  let sys : GradingSystem := {
    maxScore := 60,
    classroomWeight := 0.2,
    midtermWeight := 0.3,
    finalWeight := 0.5
  }
  let scores : StudentScores := {
    classroom := 60,
    midterm := 50,
    final := 56
  }
  calculateGrade sys scores = 55 := by
  sorry

end xiaotong_grade_l1534_153489


namespace catriona_fish_count_l1534_153459

/-- The number of fish in Catriona's aquarium -/
def total_fish (goldfish angelfish guppies tetras bettas : ℕ) : ℕ :=
  goldfish + angelfish + guppies + tetras + bettas

/-- Theorem stating the total number of fish in Catriona's aquarium -/
theorem catriona_fish_count :
  ∀ (goldfish angelfish guppies tetras bettas : ℕ),
    goldfish = 8 →
    angelfish = goldfish + 4 →
    guppies = 2 * angelfish →
    tetras = goldfish - 3 →
    bettas = tetras + 5 →
    total_fish goldfish angelfish guppies tetras bettas = 59 := by
  sorry

end catriona_fish_count_l1534_153459


namespace min_box_value_l1534_153450

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 45*x^2 + Box*x + 45) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  ∀ Box', (∀ x, (∃ a' b', a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
                 (a'*x + b') * (b'*x + a') = 45*x^2 + Box'*x + 45)) →
  Box' ≥ Box →
  Box ≥ 106 :=
by sorry

end min_box_value_l1534_153450


namespace max_value_of_expression_l1534_153496

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^10 + x^8 - 6*x^6 + 27*x^4 + 64) ≤ 1/8.38 := by
  sorry

end max_value_of_expression_l1534_153496


namespace current_age_problem_l1534_153419

theorem current_age_problem (my_age brother_age : ℕ) : 
  (my_age + 10 = 2 * (brother_age + 10)) →
  ((my_age + 10) + (brother_age + 10) = 45) →
  my_age = 20 :=
by
  sorry

end current_age_problem_l1534_153419


namespace intersection_of_M_and_N_l1534_153433

def M : Set ℝ := {x | 2 * x - 1 > 0}
def N : Set ℝ := {x | Real.sqrt x < 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1/2 < x ∧ x < 4} := by sorry

end intersection_of_M_and_N_l1534_153433


namespace jar_weight_percentage_l1534_153462

theorem jar_weight_percentage (jar_weight bean_weight : ℝ) 
  (h1 : jar_weight + 0.5 * bean_weight = 0.6 * (jar_weight + bean_weight)) : 
  jar_weight / (jar_weight + bean_weight) = 0.2 := by
sorry

end jar_weight_percentage_l1534_153462


namespace circle_center_l1534_153430

/-- The polar equation of a circle is given by ρ = √2(cos θ + sin θ).
    This theorem proves that the center of this circle is at the point (1, π/4) in polar coordinates. -/
theorem circle_center (ρ θ : ℝ) : 
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ) → 
  ∃ (r θ₀ : ℝ), r = 1 ∧ θ₀ = π / 4 ∧ 
    (∀ (x y : ℝ), x = r * Real.cos θ₀ ∧ y = r * Real.sin θ₀ → 
      (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) :=
by sorry

end circle_center_l1534_153430


namespace circle_has_most_symmetry_lines_l1534_153470

/-- Represents the number of lines of symmetry for a geometrical figure. -/
inductive SymmetryCount
  | Finite (n : ℕ)
  | Infinite

/-- Represents the geometrical figures mentioned in the problem. -/
inductive GeometricalFigure
  | Circle
  | Semicircle
  | EquilateralTriangle
  | RegularPentagon
  | Ellipse

/-- Returns the number of lines of symmetry for a given geometrical figure. -/
def symmetryLines (figure : GeometricalFigure) : SymmetryCount :=
  match figure with
  | GeometricalFigure.Circle => SymmetryCount.Infinite
  | GeometricalFigure.Semicircle => SymmetryCount.Finite 1
  | GeometricalFigure.EquilateralTriangle => SymmetryCount.Finite 3
  | GeometricalFigure.RegularPentagon => SymmetryCount.Finite 5
  | GeometricalFigure.Ellipse => SymmetryCount.Finite 2

/-- Compares two SymmetryCount values. -/
def symmetryCountLe (a b : SymmetryCount) : Prop :=
  match a, b with
  | SymmetryCount.Finite n, SymmetryCount.Finite m => n ≤ m
  | _, SymmetryCount.Infinite => True
  | SymmetryCount.Infinite, SymmetryCount.Finite _ => False

/-- States that the circle has the greatest number of lines of symmetry among the given figures. -/
theorem circle_has_most_symmetry_lines :
    ∀ (figure : GeometricalFigure),
      symmetryCountLe (symmetryLines figure) (symmetryLines GeometricalFigure.Circle) :=
  sorry

end circle_has_most_symmetry_lines_l1534_153470


namespace quadratic_sum_l1534_153461

/-- Given a quadratic function f(x) = ax^2 + bx + c where a = 2, b = -3, c = 4,
    and f(1) = 3, prove that 2a - b + c = 11 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℤ) :
  a = 2 ∧ b = -3 ∧ c = 4 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  f 1 = 3 →
  2 * a - b + c = 11 := by
sorry

end quadratic_sum_l1534_153461


namespace set_union_problem_l1534_153454

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end set_union_problem_l1534_153454


namespace quality_difference_significant_frequency_machine_A_frequency_machine_B_l1534_153452

-- Define the contingency table
def machine_A_first_class : ℕ := 150
def machine_A_second_class : ℕ := 50
def machine_B_first_class : ℕ := 120
def machine_B_second_class : ℕ := 80
def total_products : ℕ := 400

-- Define the K^2 formula
def K_squared (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value_99_percent : ℚ := 6635 / 1000

-- Theorem statement
theorem quality_difference_significant :
  K_squared machine_A_first_class machine_A_second_class
            machine_B_first_class machine_B_second_class
            total_products > critical_value_99_percent := by
  sorry

-- Frequencies of first-class products
theorem frequency_machine_A : (machine_A_first_class : ℚ) / (machine_A_first_class + machine_A_second_class) = 3 / 4 := by
  sorry

theorem frequency_machine_B : (machine_B_first_class : ℚ) / (machine_B_first_class + machine_B_second_class) = 3 / 5 := by
  sorry

end quality_difference_significant_frequency_machine_A_frequency_machine_B_l1534_153452


namespace larger_integer_proof_l1534_153421

theorem larger_integer_proof (x : ℤ) : 
  (x > 0) →  -- Ensure x is positive
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 3 → 
  4 * x = 72 :=
by sorry

end larger_integer_proof_l1534_153421


namespace greater_solution_quadratic_l1534_153474

theorem greater_solution_quadratic (x : ℝ) : 
  (x^2 + 15*x - 54 = 0) → (∃ y : ℝ, y^2 + 15*y - 54 = 0 ∧ y ≠ x) → 
  (x ≥ y ↔ x = 3) :=
sorry

end greater_solution_quadratic_l1534_153474


namespace student_marks_l1534_153426

theorem student_marks (total_marks passing_percentage failing_margin : ℕ) 
  (h1 : total_marks = 440)
  (h2 : passing_percentage = 50)
  (h3 : failing_margin = 20) : 
  (total_marks * passing_percentage / 100 - failing_margin : ℕ) = 200 :=
by sorry

end student_marks_l1534_153426


namespace speed_ratio_is_four_thirds_l1534_153434

/-- Two runners in a race where one gets a head start -/
structure Race where
  length : ℝ
  speed_a : ℝ
  speed_b : ℝ
  head_start : ℝ

/-- The race ends in a dead heat -/
def dead_heat (r : Race) : Prop :=
  r.length / r.speed_a = (r.length - r.head_start) / r.speed_b

/-- The head start is 0.25 of the race length -/
def quarter_head_start (r : Race) : Prop :=
  r.head_start = 0.25 * r.length

theorem speed_ratio_is_four_thirds (r : Race) 
  (h1 : dead_heat r) (h2 : quarter_head_start r) : 
  r.speed_a / r.speed_b = 4 / 3 := by
  sorry

end speed_ratio_is_four_thirds_l1534_153434


namespace prism_15_edges_l1534_153486

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Theorem: A prism with 15 edges has 7 faces and 10 vertices. -/
theorem prism_15_edges (p : Prism) (h : p.edges = 15) : p.faces = 7 ∧ p.vertices = 10 := by
  sorry

end prism_15_edges_l1534_153486


namespace largest_special_number_l1534_153432

def has_distinct_digits (n : ℕ) : Prop := sorry

def divisible_by_digits (n : ℕ) : Prop := sorry

def contains_digit (n : ℕ) (d : ℕ) : Prop := sorry

theorem largest_special_number :
  ∀ n : ℕ,
    has_distinct_digits n ∧
    divisible_by_digits n ∧
    contains_digit n 5 →
    n ≤ 9735 :=
by sorry

end largest_special_number_l1534_153432


namespace sodium_reduction_proof_l1534_153422

def salt_teaspoons : ℕ := 2
def initial_parmesan_ounces : ℕ := 8
def sodium_per_salt_teaspoon : ℕ := 50
def sodium_per_parmesan_ounce : ℕ := 25
def reduction_factor : ℚ := 1/3

def total_sodium (parmesan_ounces : ℕ) : ℕ :=
  salt_teaspoons * sodium_per_salt_teaspoon + parmesan_ounces * sodium_per_parmesan_ounce

def reduced_parmesan_ounces : ℕ := initial_parmesan_ounces - 4

theorem sodium_reduction_proof :
  (total_sodium initial_parmesan_ounces : ℚ) * (1 - reduction_factor) =
  (total_sodium reduced_parmesan_ounces : ℚ) := by
  sorry

end sodium_reduction_proof_l1534_153422


namespace largest_angle_right_triangle_l1534_153483

theorem largest_angle_right_triangle (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b / c = 7 / 2) : max a (max b c) = 90 := by
  sorry

end largest_angle_right_triangle_l1534_153483


namespace x_with_18_factors_l1534_153410

theorem x_with_18_factors (x : ℕ) : 
  (∃ (factors : Finset ℕ), factors.card = 18 ∧ (∀ f ∈ factors, f ∣ x)) → 
  18 ∣ x → 
  20 ∣ x → 
  x = 180 := by
sorry

end x_with_18_factors_l1534_153410


namespace subtract_decimals_l1534_153417

theorem subtract_decimals : (145.23 : ℝ) - 0.07 = 145.16 := by
  sorry

end subtract_decimals_l1534_153417


namespace fraction_value_when_y_is_three_l1534_153441

theorem fraction_value_when_y_is_three :
  let y : ℝ := 3
  (y^3 + y) / (y^2 - y) = 5 := by
sorry

end fraction_value_when_y_is_three_l1534_153441


namespace root_sum_problem_l1534_153412

theorem root_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b*(a+b) = 533 := by
sorry

end root_sum_problem_l1534_153412


namespace nine_fourth_equals_three_two_m_l1534_153499

theorem nine_fourth_equals_three_two_m (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end nine_fourth_equals_three_two_m_l1534_153499


namespace rows_sum_equal_l1534_153400

def first_row : List ℕ := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row : List ℕ := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

theorem rows_sum_equal : 
  (first_row.sum = second_row.sum + 155) := by sorry

end rows_sum_equal_l1534_153400


namespace sequence_sum_2017_l1534_153482

theorem sequence_sum_2017 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * n - 1) →
  (∀ n : ℕ, n > 0 → S n = S (n - 1) + a n) →
  a 2017 = 2 := by
  sorry

end sequence_sum_2017_l1534_153482


namespace geometric_sequence_a8_l1534_153497

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_6 = 21, prove that a_8 = 63 -/
theorem geometric_sequence_a8 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a6 : a 6 = 21) : a 8 = 63 := by
  sorry

end geometric_sequence_a8_l1534_153497


namespace sausage_pieces_l1534_153467

/-- Given a sausage with red, yellow, and green rings, prove that cutting along all rings results in 21 pieces. -/
theorem sausage_pieces (red_pieces yellow_pieces green_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11) :
  red_pieces + yellow_pieces + green_pieces - 2 = 21 :=
by sorry

end sausage_pieces_l1534_153467


namespace equilateral_triangle_perimeter_l1534_153440

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3 / 4 = 2 * s) → (3 * s = 8 * Real.sqrt 3) := by
  sorry

end equilateral_triangle_perimeter_l1534_153440


namespace sum_difference_zero_l1534_153479

theorem sum_difference_zero (x y z : ℝ) 
  (h : (2*x^2 + 8*x + 11)*(y^2 - 10*y + 29)*(3*z^2 - 18*z + 32) ≤ 60) : 
  x + y - z = 0 := by
  sorry

end sum_difference_zero_l1534_153479


namespace prob_non_red_is_half_l1534_153403

-- Define the die
def total_faces : ℕ := 10
def red_faces : ℕ := 5
def yellow_faces : ℕ := 3
def blue_faces : ℕ := 1
def green_faces : ℕ := 1

-- Define the probability of rolling a non-red face
def prob_non_red : ℚ := (yellow_faces + blue_faces + green_faces : ℚ) / total_faces

-- Theorem statement
theorem prob_non_red_is_half : prob_non_red = 1 / 2 := by
  sorry

end prob_non_red_is_half_l1534_153403


namespace floor_plus_self_eq_seventeen_point_five_l1534_153429

theorem floor_plus_self_eq_seventeen_point_five (s : ℝ) : 
  ⌊s⌋ + s = 17.5 ↔ s = 8.5 := by sorry

end floor_plus_self_eq_seventeen_point_five_l1534_153429


namespace vector_magnitude_range_function_f_range_l1534_153493

noncomputable section

def x : ℝ := sorry

-- Define vector a
def a : ℝ × ℝ := (Real.sin x + Real.cos x, Real.sqrt 2 * Real.cos x)

-- Define vector b
def b : ℝ × ℝ := (Real.cos x - Real.sin x, Real.sqrt 2 * Real.sin x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the function f(x)
def f : ℝ → ℝ := λ x => dot_product a b - magnitude a

theorem vector_magnitude_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  magnitude a ∈ Set.Icc (Real.sqrt 2) (Real.sqrt 3) :=
sorry

theorem function_f_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  f x ∈ Set.Icc (-Real.sqrt 2) (1 - Real.sqrt 3) :=
sorry

end vector_magnitude_range_function_f_range_l1534_153493


namespace only_sphere_all_circular_l1534_153425

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to determine if a view is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem: Only the Sphere has circular views from all perspectives
theorem only_sphere_all_circular :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
sorry

end only_sphere_all_circular_l1534_153425


namespace snow_probability_l1534_153468

theorem snow_probability (p : ℝ) (n : ℕ) 
  (h_p : p = 3/4) 
  (h_n : n = 4) : 
  1 - (1 - p)^n = 255/256 := by
  sorry

end snow_probability_l1534_153468


namespace robins_gum_problem_l1534_153414

theorem robins_gum_problem (initial_gum : ℝ) (total_gum : ℕ) (h1 : initial_gum = 18.0) (h2 : total_gum = 62) :
  (total_gum : ℝ) - initial_gum = 44 := by
  sorry

end robins_gum_problem_l1534_153414


namespace pages_copied_for_35_dollars_l1534_153449

/-- Given the cost of copying 5 pages is 7 cents, this theorem proves that
    the number of pages that can be copied for $35 is 2500. -/
theorem pages_copied_for_35_dollars : 
  let cost_per_5_pages : ℚ := 7 / 100  -- 7 cents in dollars
  let dollars : ℚ := 35
  let pages_per_dollar : ℚ := 5 / cost_per_5_pages
  ⌊dollars * pages_per_dollar⌋ = 2500 :=
by sorry

end pages_copied_for_35_dollars_l1534_153449


namespace linear_systems_solutions_l1534_153406

/-- Prove that the solutions to the given systems of linear equations are correct -/
theorem linear_systems_solutions :
  -- System 1
  (∃ x y : ℝ, 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℝ, 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3) :=
by sorry

end linear_systems_solutions_l1534_153406


namespace equation_solution_inequality_system_solution_l1534_153494

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 - x) / (x - 4) + 1 / (4 - x) = 1

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  2 * (x + 1) > x ∧ 1 - 2 * x ≥ (x + 7) / 2

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ x = 3 :=
sorry

theorem inequality_system_solution :
  ∃ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ -1 :=
sorry

end equation_solution_inequality_system_solution_l1534_153494


namespace remainder_of_polynomial_l1534_153420

theorem remainder_of_polynomial (n : ℤ) (k : ℤ) : 
  n = 100 * k - 1 → (n^2 + 3*n + 4) % 100 = 2 := by
sorry

end remainder_of_polynomial_l1534_153420


namespace building_shadow_length_l1534_153484

/-- Given a flagpole and a building under similar shadow-casting conditions,
    calculate the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_height_pos : 0 < building_height)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building : building_height = 20) :
  (building_height * flagpole_shadow) / flagpole_height = 50 := by
  sorry


end building_shadow_length_l1534_153484


namespace special_deck_probability_l1534_153415

/-- A deck of cards with specified properties -/
structure Deck :=
  (total : ℕ)
  (red_non_joker : ℕ)
  (black_or_joker : ℕ)

/-- The probability of drawing a red non-joker card first and a black or joker card second -/
def draw_probability (d : Deck) : ℚ :=
  (d.red_non_joker : ℚ) * (d.black_or_joker : ℚ) / ((d.total : ℚ) * (d.total - 1 : ℚ))

/-- Theorem stating the probability for the specific deck described in the problem -/
theorem special_deck_probability :
  let d := Deck.mk 60 26 40
  draw_probability d = 5 / 17 := by sorry

end special_deck_probability_l1534_153415


namespace quadratic_inequality_condition_l1534_153477

theorem quadratic_inequality_condition (x : ℝ) : 
  (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) := by sorry

end quadratic_inequality_condition_l1534_153477


namespace martha_cards_total_l1534_153404

/-- Given that Martha starts with 3 cards and receives 76 more cards,
    prove that she ends up with 79 cards in total. -/
theorem martha_cards_total : 
  let initial_cards : ℕ := 3
  let received_cards : ℕ := 76
  initial_cards + received_cards = 79 := by
  sorry

end martha_cards_total_l1534_153404


namespace pentagon_covers_half_l1534_153435

/-- Represents a tiling of a plane with large squares -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.grid_size^2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_covers_half (tiling : PlaneTiling) 
  (h1 : tiling.grid_size = 4)
  (h2 : tiling.pentagon_squares = 8) : 
  pentagon_percentage tiling = 50 := by
  sorry

end pentagon_covers_half_l1534_153435


namespace geometric_sequence_product_l1534_153481

/-- A geometric sequence is a sequence where the ratio between consecutive terms is constant. -/
def IsGeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- Given a geometric sequence b where b₂ * b₃ * b₄ = 8, prove that b₃ = 2 -/
theorem geometric_sequence_product (b : ℕ → ℝ) (h_geo : IsGeometricSequence b) 
    (h_prod : b 2 * b 3 * b 4 = 8) : b 3 = 2 := by
  sorry

end geometric_sequence_product_l1534_153481


namespace geometric_sequence_determinant_l1534_153465

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_determinant
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = 3) :
  let det := a 2 * a 8 + a 7 * a 3
  det = 18 := by
  sorry

end geometric_sequence_determinant_l1534_153465


namespace situp_rate_difference_l1534_153413

-- Define the given conditions
def diana_rate : ℕ := 4
def diana_situps : ℕ := 40
def total_situps : ℕ := 110

-- Define the theorem
theorem situp_rate_difference : ℕ := by
  -- The difference between Hani's and Diana's situp rates is 3
  sorry

end situp_rate_difference_l1534_153413


namespace phone_call_probability_l1534_153487

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.2

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering a phone call within the first four rings -/
def p_within_four_rings : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability : p_within_four_rings = 0.8 := by
  sorry

end phone_call_probability_l1534_153487


namespace percentage_problem_l1534_153424

theorem percentage_problem (x : ℝ) (h : 25 = 0.4 * x) : x = 62.5 := by
  sorry

end percentage_problem_l1534_153424


namespace geometric_sequence_property_l1534_153469

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_arithmetic : a 1 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l1534_153469


namespace vector_magnitude_l1534_153447

/-- Given two vectors a and b in ℝ² with an angle of π/3 between them,
    where a = (1, √3) and |a - 2b| = 2√3, prove that |b| = 2 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 1 ∧ a.2 = Real.sqrt 3) →  -- a = (1, √3)
  (a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) →  -- angle between a and b is π/3
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 12) →  -- |a - 2b| = 2√3
  Real.sqrt (b.1^2 + b.2^2) = 2  -- |b| = 2
:= by sorry

end vector_magnitude_l1534_153447


namespace unattainable_value_l1534_153464

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 := by
  sorry

end unattainable_value_l1534_153464


namespace gcd_power_difference_l1534_153423

theorem gcd_power_difference (a b n : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hn : n ≥ 2) (hgcd : Nat.gcd a b = 1) :
  Nat.gcd ((a^n - b^n) / (a - b)) (a - b) = Nat.gcd (a - b) n := by
  sorry

end gcd_power_difference_l1534_153423


namespace tinas_savings_l1534_153453

/-- Tina's savings problem -/
theorem tinas_savings (x : ℕ) : 
  (x + 14 + 21) - (5 + 17) = 40 → x = 27 := by
  sorry

end tinas_savings_l1534_153453


namespace sum_of_fractions_l1534_153495

theorem sum_of_fractions : 
  (1 : ℚ) / 2 + 1 / 6 + 1 / 12 + 1 / 20 + 1 / 30 + 1 / 42 = 6 / 7 := by
  sorry

end sum_of_fractions_l1534_153495


namespace bread_price_is_four_l1534_153436

/-- The price of a loaf of bread -/
def bread_price : ℝ := 4

/-- The price of a pastry -/
def pastry_price : ℝ := 2

/-- The usual number of pastries sold daily -/
def usual_pastries : ℕ := 20

/-- The usual number of loaves of bread sold daily -/
def usual_bread : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves of bread sold today -/
def today_bread : ℕ := 25

/-- The difference between today's sales and the usual daily average -/
def sales_difference : ℝ := 48

theorem bread_price_is_four :
  (today_pastries : ℝ) * pastry_price + today_bread * bread_price -
  (usual_pastries : ℝ) * pastry_price - usual_bread * bread_price = sales_difference ∧
  bread_price = 4 := by sorry

end bread_price_is_four_l1534_153436


namespace system_has_no_solution_l1534_153402

theorem system_has_no_solution :
  ¬ (∃ (x₁ x₂ x₃ x₄ : ℝ),
    (5 * x₁ + 12 * x₂ + 19 * x₃ + 25 * x₄ = 25) ∧
    (10 * x₁ + 22 * x₂ + 16 * x₃ + 39 * x₄ = 25) ∧
    (5 * x₁ + 12 * x₂ + 9 * x₃ + 25 * x₄ = 30) ∧
    (20 * x₁ + 46 * x₂ + 34 * x₃ + 89 * x₄ = 70)) :=
by
  sorry


end system_has_no_solution_l1534_153402


namespace system_solution_l1534_153460

theorem system_solution : 
  let solutions := [
    (Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2)
  ]
  ∀ (x y : ℝ), (x^2 + y^2 = 1 ∧ 4*x*y*(2*y^2 - 1) = 1) ↔ (x, y) ∈ solutions := by
sorry

end system_solution_l1534_153460


namespace choose_two_from_ten_l1534_153442

/-- The number of ways to choose 2 colors out of 10 colors -/
def choose_colors (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Choosing 2 colors out of 10 results in 45 combinations -/
theorem choose_two_from_ten :
  choose_colors 10 2 = 45 := by
  sorry

end choose_two_from_ten_l1534_153442


namespace one_third_green_faces_iff_three_l1534_153431

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of green faces on unit cubes after cutting a large painted cube -/
def green_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces on all unit cubes after cutting -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- Theorem stating that exactly one-third of faces are green iff n = 3 -/
theorem one_third_green_faces_iff_three (c : Cube n) :
  3 * green_faces c = total_faces c ↔ n = 3 :=
sorry

end one_third_green_faces_iff_three_l1534_153431


namespace average_marks_l1534_153401

theorem average_marks (n : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 86) :
  ((n - 1) * avg_five + sixth_mark) / n = 76 := by
  sorry

end average_marks_l1534_153401


namespace spinner_probability_l1534_153407

-- Define the spinners
def spinner_C : Finset ℕ := {1, 3, 5, 7}
def spinner_D : Finset ℕ := {2, 4, 6}

-- Define the probability space
def Ω : Finset (ℕ × ℕ) := spinner_C.product spinner_D

-- Define the event of sum being divisible by 3
def E : Finset (ℕ × ℕ) := Ω.filter (λ p => (p.1 + p.2) % 3 = 0)

theorem spinner_probability : 
  (Finset.card E : ℚ) / (Finset.card Ω : ℚ) = 1 / 4 := by sorry

end spinner_probability_l1534_153407


namespace chess_tournament_games_l1534_153409

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 := by sorry

end chess_tournament_games_l1534_153409


namespace ship_length_in_emily_steps_l1534_153456

theorem ship_length_in_emily_steps :
  ∀ (emily_speed ship_speed : ℝ) (emily_steps_forward emily_steps_backward : ℕ),
    emily_speed > ship_speed →
    emily_steps_forward = 300 →
    emily_steps_backward = 60 →
    ship_speed > 0 →
    ∃ (ship_length : ℝ),
      ship_length = emily_steps_forward * emily_speed / (emily_speed + ship_speed) +
                    emily_steps_backward * emily_speed / (emily_speed - ship_speed) ∧
      ship_length = 100 := by
  sorry

end ship_length_in_emily_steps_l1534_153456
