import Mathlib

namespace NUMINAMATH_CALUDE_percentage_composition_l806_80673

theorem percentage_composition (F S T : ℝ) 
  (h1 : F = 0.20 * S) 
  (h2 : S = 0.25 * T) : 
  F = 0.05 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_composition_l806_80673


namespace NUMINAMATH_CALUDE_age_sum_proof_l806_80684

/-- Given the age relationship between Michael and Emily, prove that the sum of their current ages is 32. -/
theorem age_sum_proof (M E : ℚ) : 
  M = E + 9 ∧ 
  M + 5 = 3 * (E - 3) → 
  M + E = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l806_80684


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l806_80601

theorem prime_sum_theorem (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧  -- p, q, r, s are primes
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  Prime (p + q + r + s) ∧  -- their sum is prime
  ∃ a, p^2 + q*r = a^2 ∧  -- p² + qr is a perfect square
  ∃ b, p^2 + q*s = b^2  -- p² + qs is a perfect square
  → p + q + r + s = 23 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l806_80601


namespace NUMINAMATH_CALUDE_shaded_area_concentric_circles_l806_80650

theorem shaded_area_concentric_circles 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ > 0) 
  (h₂ : r₂ > r₁) 
  (h₃ : r₁ / (r₂ - r₁) = 1 / 2) 
  (h₄ : r₂ = 9) : 
  π * r₂^2 - π * r₁^2 = 72 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_concentric_circles_l806_80650


namespace NUMINAMATH_CALUDE_kangaroo_koala_ratio_l806_80685

theorem kangaroo_koala_ratio :
  let total_animals : ℕ := 216
  let num_kangaroos : ℕ := 180
  let num_koalas : ℕ := total_animals - num_kangaroos
  num_kangaroos / num_koalas = 5 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_koala_ratio_l806_80685


namespace NUMINAMATH_CALUDE_worker_net_income_proof_l806_80687

/-- Calculates the net income after tax for a tax resident worker --/
def netIncomeAfterTax (grossIncome : ℝ) (taxRate : ℝ) : ℝ :=
  grossIncome * (1 - taxRate)

/-- Proves that the net income after tax for a worker credited with 45000 and a 13% tax rate is 39150 --/
theorem worker_net_income_proof :
  let grossIncome : ℝ := 45000
  let taxRate : ℝ := 0.13
  netIncomeAfterTax grossIncome taxRate = 39150 := by
sorry

#eval netIncomeAfterTax 45000 0.13

end NUMINAMATH_CALUDE_worker_net_income_proof_l806_80687


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l806_80619

theorem simplify_absolute_value : |(-4)^2 - 3^2 + 2| = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l806_80619


namespace NUMINAMATH_CALUDE_parsley_sprigs_theorem_l806_80694

/-- Calculates the number of parsley sprigs left after decorating plates -/
def sprigs_left (initial_sprigs : ℕ) (whole_sprig_plates : ℕ) (half_sprig_plates : ℕ) : ℕ :=
  initial_sprigs - (whole_sprig_plates + (half_sprig_plates / 2))

/-- Proves that given the specific conditions, 11 sprigs are left -/
theorem parsley_sprigs_theorem :
  sprigs_left 25 8 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_parsley_sprigs_theorem_l806_80694


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l806_80661

theorem probability_at_least_one_woman (total : ℕ) (men women selected : ℕ) 
  (h_total : total = men + women)
  (h_men : men = 6)
  (h_women : women = 4)
  (h_selected : selected = 3) :
  1 - (Nat.choose men selected : ℚ) / (Nat.choose total selected) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l806_80661


namespace NUMINAMATH_CALUDE_fraction_transformation_impossibility_l806_80653

theorem fraction_transformation_impossibility : ¬∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_impossibility_l806_80653


namespace NUMINAMATH_CALUDE_complex_product_sum_l806_80643

theorem complex_product_sum (a b : ℝ) : (1 + Complex.I) * (1 - Complex.I) = a + b * Complex.I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l806_80643


namespace NUMINAMATH_CALUDE_sum_equals_seven_eighths_l806_80672

theorem sum_equals_seven_eighths : 
  let original_sum := 1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64
  let removed_terms := 1/16 + 1/32 + 1/64
  let remaining_terms := original_sum - removed_terms
  remaining_terms = 7/8 := by sorry

end NUMINAMATH_CALUDE_sum_equals_seven_eighths_l806_80672


namespace NUMINAMATH_CALUDE_deck_size_proof_l806_80612

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/5 →
  r + b = 24 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l806_80612


namespace NUMINAMATH_CALUDE_hill_climbing_speeds_l806_80608

theorem hill_climbing_speeds (distance : ℝ) (ascending_time descending_time : ℝ) 
  (h1 : ascending_time = 3)
  (h2 : distance / ascending_time = 2.5)
  (h3 : (2 * distance) / (ascending_time + descending_time) = 3) :
  distance / descending_time = 3.75 := by sorry

end NUMINAMATH_CALUDE_hill_climbing_speeds_l806_80608


namespace NUMINAMATH_CALUDE_forest_to_street_ratio_l806_80610

/-- The ratio of forest area to street area is 3:1 -/
theorem forest_to_street_ratio : 
  ∀ (street_side_length : ℝ) (trees_per_sq_meter : ℝ) (total_trees : ℝ),
  street_side_length = 100 →
  trees_per_sq_meter = 4 →
  total_trees = 120000 →
  (total_trees / trees_per_sq_meter) / (street_side_length ^ 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_forest_to_street_ratio_l806_80610


namespace NUMINAMATH_CALUDE_largest_solution_inverse_power_twelve_l806_80628

-- Define the logarithmic equation
def log_equation (x : ℝ) : Prop :=
  Real.log 10 / Real.log (10 * x^3) + Real.log 10 / Real.log (100 * x^4) = -1

-- Define the set of solutions
def solution_set : Set ℝ :=
  {x | x > 0 ∧ log_equation x}

-- State the theorem
theorem largest_solution_inverse_power_twelve :
  ∃ x ∈ solution_set, ∀ y ∈ solution_set, y ≤ x → (1 : ℝ) / x^12 = 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_inverse_power_twelve_l806_80628


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l806_80699

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 9 → a * b = 1800 → Nat.lcm a b = 200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l806_80699


namespace NUMINAMATH_CALUDE_ice_cream_volume_l806_80611

/-- The volume of ice cream in a cone with hemisphere and cylindrical topping -/
theorem ice_cream_volume (h_cone r_cone h_cylinder : ℝ) 
  (h_cone_pos : 0 < h_cone)
  (r_cone_pos : 0 < r_cone)
  (h_cylinder_pos : 0 < h_cylinder)
  (h_cone_val : h_cone = 12)
  (r_cone_val : r_cone = 3)
  (h_cylinder_val : h_cylinder = 2) :
  (1/3 * π * r_cone^2 * h_cone) +  -- Volume of cone
  (2/3 * π * r_cone^3) +           -- Volume of hemisphere
  (π * r_cone^2 * h_cylinder) =    -- Volume of cylinder
  72 * π := by
sorry


end NUMINAMATH_CALUDE_ice_cream_volume_l806_80611


namespace NUMINAMATH_CALUDE_polynomial_simplification_l806_80654

theorem polynomial_simplification (x : ℝ) : 
  (7 * x^12 + 2 * x^10 + x^9) + (3 * x^11 + x^10 + 6 * x^9 + 5 * x^7) + (x^12 + 4 * x^10 + 2 * x^9 + x^3) = 
  8 * x^12 + 3 * x^11 + 7 * x^10 + 9 * x^9 + 5 * x^7 + x^3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l806_80654


namespace NUMINAMATH_CALUDE_hyperbola_equation_l806_80659

/-- Given two hyperbolas with the same asymptotes and a specific focus, prove the equation of one hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of C₁
  (∀ x y : ℝ, x^2/4 - y^2/16 = 1) →    -- Equation of C₂
  (b/a = 2) →                          -- Same asymptotes condition
  (a^2 + b^2 = 5) →                    -- Right focus condition
  (∀ x y : ℝ, x^2 - y^2/4 = 1) :=      -- Conclusion: Equation of C₁
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l806_80659


namespace NUMINAMATH_CALUDE_rachel_coloring_books_l806_80677

/-- The number of pictures remaining to be colored given the number of pictures in three coloring books and the number of pictures already colored. -/
def remaining_pictures (book1 book2 book3 colored : ℕ) : ℕ :=
  book1 + book2 + book3 - colored

/-- Theorem stating that given the specific numbers from the problem, the remaining pictures to be colored is 56. -/
theorem rachel_coloring_books : remaining_pictures 23 32 45 44 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rachel_coloring_books_l806_80677


namespace NUMINAMATH_CALUDE_horner_method_first_step_l806_80666

def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

def horner_first_step (a₅ a₄ : ℝ) (x : ℝ) : ℝ := a₅ * x + a₄

theorem horner_method_first_step :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horner_method_first_step_l806_80666


namespace NUMINAMATH_CALUDE_ratio_problem_l806_80624

theorem ratio_problem (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l806_80624


namespace NUMINAMATH_CALUDE_matrix_power_negative_identity_l806_80617

open Matrix

theorem matrix_power_negative_identity 
  (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (n : ℕ) 
  (hn : n ≠ 0) 
  (hA : A ^ n = -1 • (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -1 • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ 
  A ^ 3 = -1 • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
by sorry

end NUMINAMATH_CALUDE_matrix_power_negative_identity_l806_80617


namespace NUMINAMATH_CALUDE_fourth_root_of_2560000_l806_80603

theorem fourth_root_of_2560000 : Real.sqrt (Real.sqrt 2560000) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_2560000_l806_80603


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l806_80629

/-- Given a cistern that fills in 8 hours without a leak and 9 hours with a leak,
    the time it takes for the leak to empty a full cistern is 72 hours. -/
theorem cistern_leak_emptying_time :
  ∀ (fill_rate_no_leak : ℝ) (fill_rate_with_leak : ℝ) (leak_rate : ℝ),
    fill_rate_no_leak = 1 / 8 →
    fill_rate_with_leak = 1 / 9 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 / leak_rate : ℝ) = 72 := by
  sorry


end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l806_80629


namespace NUMINAMATH_CALUDE_consecutive_integer_products_sum_l806_80634

theorem consecutive_integer_products_sum : 
  ∃ (a b c x y z w : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (y = x + 1) ∧ 
    (z = y + 1) ∧ 
    (w = z + 1) ∧ 
    (a * b * c = 924) ∧ 
    (x * y * z * w = 924) ∧ 
    (a + b + c + x + y + z + w = 75) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_products_sum_l806_80634


namespace NUMINAMATH_CALUDE_minute_hand_angle_after_160_minutes_l806_80627

/-- Represents the movement of a clock's hour hand in minutes -/
def hourHandMovement : ℕ := 2 * 60 + 40

/-- The angle turned by the minute hand in degrees -/
def minuteHandAngle : ℤ := -960

/-- Proves that when the hour hand of a clock has moved for 2 hours and 40 minutes,
    the angle turned by the minute hand is -960°, given that clock hands always rotate clockwise -/
theorem minute_hand_angle_after_160_minutes :
  minuteHandAngle = -((hourHandMovement / 60 : ℕ) * 360 + (hourHandMovement % 60 : ℕ) * 6) :=
by sorry

end NUMINAMATH_CALUDE_minute_hand_angle_after_160_minutes_l806_80627


namespace NUMINAMATH_CALUDE_parabola_equation_from_ellipse_focus_l806_80638

/-- The standard equation of a parabola with its focus at the right focus of the ellipse x^2/3 + y^2 = 1 -/
theorem parabola_equation_from_ellipse_focus : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 3 + y^2 = 1 ∧ x > 0) → 
    (∀ (u v : ℝ), v^2 = 4 * Real.sqrt 2 * u ↔ 
      (u - x)^2 + v^2 = (u - a)^2 + v^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_ellipse_focus_l806_80638


namespace NUMINAMATH_CALUDE_number_of_lineups_is_4290_l806_80676

/-- Represents the total number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- Represents the number of players who refuse to play together -/
def incompatible_players : ℕ := 2

/-- Calculates the number of possible starting lineups -/
def number_of_lineups : ℕ :=
  let remaining_players := total_players - incompatible_players
  Nat.choose remaining_players (lineup_size - 1) * 2 +
  Nat.choose remaining_players lineup_size

/-- Theorem stating that the number of possible lineups is 4290 -/
theorem number_of_lineups_is_4290 :
  number_of_lineups = 4290 := by sorry

end NUMINAMATH_CALUDE_number_of_lineups_is_4290_l806_80676


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_300_l806_80626

theorem least_integer_greater_than_sqrt_300 : ∃ n : ℕ, n > ⌊Real.sqrt 300⌋ ∧ ∀ m : ℕ, m > ⌊Real.sqrt 300⌋ → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_300_l806_80626


namespace NUMINAMATH_CALUDE_sqrt_simplification_l806_80602

theorem sqrt_simplification : 3 * Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l806_80602


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l806_80682

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^5 + β^5 + γ^5 = 47.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l806_80682


namespace NUMINAMATH_CALUDE_at_least_one_false_l806_80693

theorem at_least_one_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_false_l806_80693


namespace NUMINAMATH_CALUDE_school_election_votes_l806_80625

/-- Represents the total number of votes in a school election --/
def total_votes : ℕ := 180

/-- Represents Brenda's share of the total votes --/
def brenda_fraction : ℚ := 4 / 15

/-- Represents the number of votes Brenda received --/
def brenda_votes : ℕ := 48

/-- Represents the number of votes Colby received --/
def colby_votes : ℕ := 35

/-- Theorem stating that given the conditions, the total number of votes is 180 --/
theorem school_election_votes : 
  (brenda_fraction * total_votes = brenda_votes) ∧ 
  (colby_votes < total_votes) ∧ 
  (brenda_votes + colby_votes < total_votes) :=
sorry


end NUMINAMATH_CALUDE_school_election_votes_l806_80625


namespace NUMINAMATH_CALUDE_prime_sum_equality_l806_80683

theorem prime_sum_equality (p q n : ℕ) : 
  Prime p → Prime q → 0 < n → 
  p * (p + 3) + q * (q + 3) = n * (n + 3) → 
  ((p = 2 ∧ q = 3 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 2 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 7 ∧ n = 8) ∨ 
   (p = 7 ∧ q = 3 ∧ n = 8)) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l806_80683


namespace NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l806_80605

/-- The minimum amount spent on boxes for packaging a collection --/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ) (box_width : ℝ) (box_height : ℝ)
  (cost_per_box : ℝ) (total_collection_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.40)
  (h5 : total_collection_volume = 2160000) :
  ⌈total_collection_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 180 := by
  sorry

#check minimum_amount_spent_on_boxes

end NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l806_80605


namespace NUMINAMATH_CALUDE_alpha_value_at_negative_six_l806_80613

/-- Given that α is inversely proportional to β², prove that α = 1/3 when β = -6,
    given the condition that α = 3 when β = 2. -/
theorem alpha_value_at_negative_six (α β : ℝ) (h : ∃ k, ∀ β ≠ 0, α = k / β^2) 
    (h_condition : α = 3 ∧ β = 2) : 
    (β = -6) → (α = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_at_negative_six_l806_80613


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l806_80689

/-- The shortest distance from a circle to a line --/
theorem shortest_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y + 3)^2 = 9}
  let line := {(x, y) : ℝ × ℝ | y = x}
  (∃ (d : ℝ), d = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      d ≤ Real.sqrt ((p.1 - p.2)^2 / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l806_80689


namespace NUMINAMATH_CALUDE_root_equation_problem_l806_80678

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l806_80678


namespace NUMINAMATH_CALUDE_quadratic_residue_characterization_l806_80631

theorem quadratic_residue_characterization (a b c : ℕ+) :
  (∀ (p : ℕ) (hp : Prime p) (n : ℤ), 
    (∃ (m : ℤ), n ≡ m^2 [ZMOD p]) → 
    (∃ (k : ℤ), (a.val : ℤ) * n^2 + (b.val : ℤ) * n + (c.val : ℤ) ≡ k^2 [ZMOD p])) ↔
  (∃ (d e : ℤ), (a : ℤ) = d^2 ∧ (b : ℤ) = 2*d*e ∧ (c : ℤ) = e^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_residue_characterization_l806_80631


namespace NUMINAMATH_CALUDE_total_jumps_calculation_total_jumps_is_4411_l806_80663

/-- Calculate the total number of jumps made by Rupert and Ronald throughout the week. -/
theorem total_jumps_calculation : ℕ := by
  -- Define the number of jumps for Ronald on Monday
  let ronald_monday : ℕ := 157

  -- Define Rupert's jumps on Monday relative to Ronald's
  let rupert_monday : ℕ := ronald_monday + 86

  -- Define Ronald's jumps on Tuesday
  let ronald_tuesday : ℕ := 193

  -- Define Rupert's jumps on Tuesday
  let rupert_tuesday : ℕ := rupert_monday - 35

  -- Define the constant decrease rate from Thursday to Sunday
  let daily_decrease : ℕ := 20

  -- Calculate total jumps
  let total_jumps : ℕ := 
    -- Monday
    ronald_monday + rupert_monday +
    -- Tuesday
    ronald_tuesday + rupert_tuesday +
    -- Wednesday (doubled from Tuesday)
    2 * ronald_tuesday + 2 * rupert_tuesday +
    -- Thursday to Sunday (4 days with constant decrease)
    (2 * ronald_tuesday - daily_decrease) + (2 * rupert_tuesday - daily_decrease) +
    (2 * ronald_tuesday - 2 * daily_decrease) + (2 * rupert_tuesday - 2 * daily_decrease) +
    (2 * ronald_tuesday - 3 * daily_decrease) + (2 * rupert_tuesday - 3 * daily_decrease) +
    (2 * ronald_tuesday - 4 * daily_decrease) + (2 * rupert_tuesday - 4 * daily_decrease)

  exact total_jumps

/-- Prove that the total number of jumps is 4411 -/
theorem total_jumps_is_4411 : total_jumps_calculation = 4411 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_calculation_total_jumps_is_4411_l806_80663


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l806_80651

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 5 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l806_80651


namespace NUMINAMATH_CALUDE_initial_profit_percentage_l806_80698

/-- Proves that the initial profit percentage is 5% given the conditions of the problem -/
theorem initial_profit_percentage (cost_price selling_price : ℝ) : 
  cost_price = 1000 →
  (0.95 * cost_price) * 1.1 = selling_price - 5 →
  (selling_price - cost_price) / cost_price = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_l806_80698


namespace NUMINAMATH_CALUDE_hyperbola_equation_l806_80620

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := fun x y => (x^2 / a^2) - (y^2 / b^2) = 1

/-- The focus of a hyperbola -/
def Focus := ℝ × ℝ

/-- Theorem stating the equation of a specific hyperbola -/
theorem hyperbola_equation (H : Hyperbola) (F : Focus) :
  H.equation = fun x y => (y^2 / 12) - (x^2 / 24) = 1 ↔
    F = (0, 6) ∧
    ∃ (K : Hyperbola), K.a^2 = 2 ∧ K.b^2 = 1 ∧
      (∀ x y, H.equation x y ↔ K.equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l806_80620


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l806_80616

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l806_80616


namespace NUMINAMATH_CALUDE_sum_a_b_eq_neg_four_l806_80670

theorem sum_a_b_eq_neg_four (a b : ℝ) (h : |1 - 2*a + b| + 2*a = -a^2 - 1) : 
  a + b = -4 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_eq_neg_four_l806_80670


namespace NUMINAMATH_CALUDE_factor_expression_l806_80669

theorem factor_expression (x : ℝ) : 270 * x^3 - 90 * x^2 + 18 * x = 18 * x * (15 * x^2 - 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l806_80669


namespace NUMINAMATH_CALUDE_cuboid_volume_transformation_l806_80648

theorem cuboid_volume_transformation (V : ℝ) (h : V = 343) : 
  let s := V^(1/3)
  let L := 3 * s
  let W := 1.5 * s
  let H := 2.5 * s
  L * W * H = 38587.5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_transformation_l806_80648


namespace NUMINAMATH_CALUDE_triangle_rectangle_perimeter_l806_80658

theorem triangle_rectangle_perimeter (d : ℕ) : 
  ∀ (t w : ℝ),
  t > 0 ∧ w > 0 →  -- positive sides
  3 * t - (6 * w) = 2016 →  -- perimeter difference
  t = 2 * w + d →  -- side length difference
  d = 672 ∧ ∀ (x : ℕ), x ≠ 672 → x ≠ d :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_perimeter_l806_80658


namespace NUMINAMATH_CALUDE_delegates_without_badges_l806_80671

theorem delegates_without_badges (total : ℕ) (pre_printed : ℚ) (break_fraction : ℚ) (hand_written : ℚ) 
  (h_total : total = 100)
  (h_pre_printed : pre_printed = 1/5)
  (h_break : break_fraction = 3/7)
  (h_hand_written : hand_written = 2/9) :
  ↑total - (↑total * pre_printed).floor - 
  ((↑total - (↑total * pre_printed).floor) * break_fraction).floor - 
  (((↑total - (↑total * pre_printed).floor) - ((↑total - (↑total * pre_printed).floor) * break_fraction).floor) * hand_written).floor = 36 :=
by sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l806_80671


namespace NUMINAMATH_CALUDE_total_waiting_time_l806_80600

def days_first_appointment : ℕ := 4
def days_second_appointment : ℕ := 20
def weeks_for_effectiveness : ℕ := 2
def days_per_week : ℕ := 7

theorem total_waiting_time :
  days_first_appointment + days_second_appointment + (weeks_for_effectiveness * days_per_week) = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_waiting_time_l806_80600


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l806_80640

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l806_80640


namespace NUMINAMATH_CALUDE_spherical_segment_surface_area_equals_circle_area_l806_80688

/-- Given a spherical segment with radius R and height H, and a circle with radius b
    where b² = 2RH, the surface area of the spherical segment (2πRH) is equal to
    the area of the circle (πb²). -/
theorem spherical_segment_surface_area_equals_circle_area
  (R H b : ℝ) (h : b^2 = 2 * R * H) :
  2 * Real.pi * R * H = Real.pi * b^2 := by
  sorry

end NUMINAMATH_CALUDE_spherical_segment_surface_area_equals_circle_area_l806_80688


namespace NUMINAMATH_CALUDE_pi_minus_three_zero_plus_half_inverse_equals_three_l806_80664

theorem pi_minus_three_zero_plus_half_inverse_equals_three :
  (Real.pi - 3) ^ (0 : ℕ) + (1 / 2) ^ (-1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pi_minus_three_zero_plus_half_inverse_equals_three_l806_80664


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l806_80623

def number_of_cages : ℕ := 15
def empty_cages : ℕ := 3
def number_of_chickens : ℕ := 3
def number_of_dogs : ℕ := 3
def number_of_cats : ℕ := 6

def arrangement_count : ℕ := Nat.choose number_of_cages empty_cages * 
                              Nat.factorial 3 * 
                              Nat.factorial number_of_chickens * 
                              Nat.factorial number_of_dogs * 
                              Nat.factorial number_of_cats

theorem animal_arrangement_count : arrangement_count = 70761600 := by
  sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l806_80623


namespace NUMINAMATH_CALUDE_card_arrangement_possible_l806_80662

def initial_sequence : List ℕ := [7, 8, 9, 4, 5, 6, 1, 2, 3]
def final_sequence : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def reverse_sublist (l : List α) (start finish : ℕ) : List α :=
  (l.take start) ++ (l.drop start |>.take (finish - start + 1) |>.reverse) ++ (l.drop (finish + 1))

def can_transform (l : List ℕ) : Prop :=
  ∃ (s1 f1 s2 f2 s3 f3 : ℕ),
    reverse_sublist (reverse_sublist (reverse_sublist l s1 f1) s2 f2) s3 f3 = final_sequence

theorem card_arrangement_possible :
  can_transform initial_sequence :=
sorry

end NUMINAMATH_CALUDE_card_arrangement_possible_l806_80662


namespace NUMINAMATH_CALUDE_possible_values_of_a_l806_80660

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l806_80660


namespace NUMINAMATH_CALUDE_number_equation_l806_80641

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l806_80641


namespace NUMINAMATH_CALUDE_wheel_probability_l806_80668

theorem wheel_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l806_80668


namespace NUMINAMATH_CALUDE_chloe_points_per_treasure_l806_80621

/-- The number of treasures Chloe found on the first level -/
def treasures_level1 : ℕ := 6

/-- The number of treasures Chloe found on the second level -/
def treasures_level2 : ℕ := 3

/-- Chloe's total score -/
def total_score : ℕ := 81

/-- The number of points Chloe scores for each treasure -/
def points_per_treasure : ℕ := total_score / (treasures_level1 + treasures_level2)

theorem chloe_points_per_treasure :
  points_per_treasure = 9 := by
  sorry

end NUMINAMATH_CALUDE_chloe_points_per_treasure_l806_80621


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_quadrilateral_relation_l806_80609

/-- A quadrilateral inscribed in one circle and circumscribed about another -/
structure InscribedCircumscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  r : ℝ  -- radius of the inscribed circle
  d : ℝ  -- distance between the centers of the circles
  R_pos : 0 < R
  r_pos : 0 < r
  d_pos : 0 < d
  d_lt_R : d < R

/-- The relationship between R, r, and d for an inscribed-circumscribed quadrilateral -/
theorem inscribed_circumscribed_quadrilateral_relation 
  (q : InscribedCircumscribedQuadrilateral) : 
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_quadrilateral_relation_l806_80609


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l806_80680

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l806_80680


namespace NUMINAMATH_CALUDE_pigeonhole_principle_balls_l806_80686

theorem pigeonhole_principle_balls (red yellow blue : ℕ) :
  red > 0 ∧ yellow > 0 ∧ blue > 0 →
  ∃ n : ℕ, n = 4 ∧
    ∀ k : ℕ, k < n →
      ∃ f : Fin k → Fin 3,
        ∀ i j : Fin k, i ≠ j → f i = f j →
          ∃ m : ℕ, m ≥ n ∧
            ∀ g : Fin m → Fin 3,
              ∃ i j : Fin m, i ≠ j ∧ g i = g j :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_balls_l806_80686


namespace NUMINAMATH_CALUDE_only_statement3_correct_l806_80649

-- Define even and odd functions
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the statements
def Statement1 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∃ y, f 0 = y
def Statement2 : Prop := ∀ f : ℝ → ℝ, OddFunction f → f 0 = 0
def Statement3 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∀ x, f x = f (-x)
def Statement4 : Prop := ∀ f : ℝ → ℝ, (EvenFunction f ∧ OddFunction f) → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem only_statement3_correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement3_correct_l806_80649


namespace NUMINAMATH_CALUDE_surtido_criterion_l806_80644

def sum_of_digits (A : ℕ) : ℕ := sorry

def is_sum_of_digits (A n : ℕ) : Prop := sorry

def is_surtido (A : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ sum_of_digits A → is_sum_of_digits A k

theorem surtido_criterion (A : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → is_sum_of_digits A k) → is_surtido A := by sorry

end NUMINAMATH_CALUDE_surtido_criterion_l806_80644


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l806_80645

theorem sqrt_difference_equality : 
  Real.sqrt (121 + 81) - Real.sqrt (49 - 36) = Real.sqrt 202 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l806_80645


namespace NUMINAMATH_CALUDE_deduce_day_from_statements_l806_80637

structure Animal where
  name : String
  lying_days : Finset Nat

def day_of_week (d : Nat) : Nat :=
  d % 7

theorem deduce_day_from_statements
  (lion unicorn : Animal)
  (today yesterday : Nat)
  (h_lion_statement : day_of_week yesterday ∈ lion.lying_days)
  (h_unicorn_statement : day_of_week yesterday ∈ unicorn.lying_days)
  (h_common_lying_day : ∃! d, d ∈ lion.lying_days ∧ d ∈ unicorn.lying_days)
  (h_today_yesterday : day_of_week today = (day_of_week yesterday + 1) % 7) :
  ∃ (common_day : Nat),
    day_of_week yesterday = common_day ∧
    common_day ∈ lion.lying_days ∧
    common_day ∈ unicorn.lying_days ∧
    day_of_week today = (common_day + 1) % 7 :=
by sorry

end NUMINAMATH_CALUDE_deduce_day_from_statements_l806_80637


namespace NUMINAMATH_CALUDE_solution_pairs_count_l806_80656

theorem solution_pairs_count : 
  let equation := λ (x y : ℕ) => 4 * x + 7 * y = 600
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => equation p.1 p.2) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l806_80656


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l806_80635

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_wrt_x_axis (a - 1, 5) (2, b - 1) →
  (a + b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l806_80635


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l806_80665

theorem lawn_mowing_earnings 
  (lawns_mowed : ℕ) 
  (initial_savings : ℕ) 
  (total_after_mowing : ℕ) 
  (h1 : lawns_mowed = 5)
  (h2 : initial_savings = 7)
  (h3 : total_after_mowing = 47) :
  (total_after_mowing - initial_savings) / lawns_mowed = 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l806_80665


namespace NUMINAMATH_CALUDE_marathon_time_proof_l806_80630

theorem marathon_time_proof (dean_time jake_time micah_time : ℝ) : 
  dean_time = 9 →
  micah_time = (2/3) * dean_time →
  jake_time = micah_time + (1/3) * micah_time →
  micah_time + dean_time + jake_time = 23 := by
sorry

end NUMINAMATH_CALUDE_marathon_time_proof_l806_80630


namespace NUMINAMATH_CALUDE_sum_of_digits_greater_than_4_l806_80692

def digits_of_735 : List Nat := [7, 3, 5]

def is_valid_card (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_of_digits_greater_than_4 :
  (∀ d ∈ digits_of_735, is_valid_card d) →
  (List.sum (digits_of_735.filter (λ x => x > 4))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_greater_than_4_l806_80692


namespace NUMINAMATH_CALUDE_min_value_theorem_l806_80655

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 10) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (w : ℝ), w = x^2 + y^2 + z^2 + x^2*y → w ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l806_80655


namespace NUMINAMATH_CALUDE_remainder_theorem_l806_80642

theorem remainder_theorem (n : ℕ) : (2 * n) % 4 = 2 → n % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l806_80642


namespace NUMINAMATH_CALUDE_constant_phi_is_cone_l806_80674

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A cone in 3D space -/
def Cone : Set SphericalCoord := sorry

/-- The shape described by φ = c in spherical coordinates -/
def ConstantPhiShape (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

/-- Theorem stating that the shape described by φ = c is a cone -/
theorem constant_phi_is_cone (c : ℝ) :
  ConstantPhiShape c = Cone := by sorry

end NUMINAMATH_CALUDE_constant_phi_is_cone_l806_80674


namespace NUMINAMATH_CALUDE_andrews_piggy_bank_donation_l806_80639

/-- Calculates the amount Andrew donated from his piggy bank to the homeless shelter --/
theorem andrews_piggy_bank_donation
  (total_earnings : ℕ)
  (ingredient_cost : ℕ)
  (total_shelter_donation : ℕ)
  (h1 : total_earnings = 400)
  (h2 : ingredient_cost = 100)
  (h3 : total_shelter_donation = 160) :
  total_shelter_donation - ((total_earnings - ingredient_cost) / 2) = 10 := by
  sorry

#check andrews_piggy_bank_donation

end NUMINAMATH_CALUDE_andrews_piggy_bank_donation_l806_80639


namespace NUMINAMATH_CALUDE_peanut_eating_interval_l806_80604

/-- Proves that given a flight duration of 2 hours and 4 bags of peanuts with 30 peanuts each,
    if all peanuts are consumed at equally spaced intervals during the flight,
    the time between eating each peanut is 1 minute. -/
theorem peanut_eating_interval (flight_duration : ℕ) (bags : ℕ) (peanuts_per_bag : ℕ) :
  flight_duration = 2 →
  bags = 4 →
  peanuts_per_bag = 30 →
  (flight_duration * 60) / (bags * peanuts_per_bag) = 1 := by
  sorry

#check peanut_eating_interval

end NUMINAMATH_CALUDE_peanut_eating_interval_l806_80604


namespace NUMINAMATH_CALUDE_salary_increase_l806_80675

theorem salary_increase (S : ℝ) (savings_rate_year1 savings_rate_year2 savings_ratio : ℝ) :
  savings_rate_year1 = 0.10 →
  savings_rate_year2 = 0.06 →
  savings_ratio = 0.6599999999999999 →
  ∃ (P : ℝ), 
    savings_rate_year2 * S * (1 + P / 100) = savings_ratio * (savings_rate_year1 * S) ∧
    P = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l806_80675


namespace NUMINAMATH_CALUDE_football_team_size_l806_80615

theorem football_team_size :
  ∀ (P : ℕ),
  (49 : ℕ) ≤ P →  -- There are at least 49 throwers
  (63 : ℕ) ≤ P →  -- There are at least 63 right-handed players
  (P - 49) % 3 = 0 →  -- The non-throwers can be divided into thirds
  63 = 49 + (2 * (P - 49) / 3) →  -- Right-handed players equation
  P = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_size_l806_80615


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l806_80636

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l806_80636


namespace NUMINAMATH_CALUDE_remainder_problem_l806_80614

theorem remainder_problem (M : ℕ) (h1 : M % 24 = 13) (h2 : M = 3024) : M % 1821 = 1203 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l806_80614


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l806_80647

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 1 / 2) : 
  Real.cos (2 * θ) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l806_80647


namespace NUMINAMATH_CALUDE_solve_for_a_l806_80633

-- Define the equations as functions of x
def eq1 (x : ℝ) : Prop := 6 * (x + 8) = 18 * x
def eq2 (a x : ℝ) : Prop := 6 * x - 2 * (a - x) = 2 * a + x

-- State the theorem
theorem solve_for_a : ∃ (a : ℝ), ∃ (x : ℝ), eq1 x ∧ eq2 a x ∧ a = 7 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l806_80633


namespace NUMINAMATH_CALUDE_conclusions_C_and_D_are_incorrect_l806_80657

-- Define the set of elements
def S : Set Char := {'a', 'b', 'c', 'd', 'e', 'f'}

-- Define the subset relation
def isSubset (A B : Set Char) : Prop := A ⊆ B

-- Define the strict subset relation
def isStrictSubset (A B : Set Char) : Prop := A ⊂ B

-- Define the number of sets satisfying the condition in A
def numSets : ℕ := 7

-- Define the quadratic equation in B
def hasOnePositiveOneNegativeRoot (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + x + a = 0 ∧ y^2 + y + a = 0

-- Define the condition in C
def conditionC (a b : ℝ) : Prop := a ≠ 0 → a * b ≠ 0

-- Define the solution set in D
def solutionSetD : Set ℝ := {x | x < 1}

theorem conclusions_C_and_D_are_incorrect :
  (¬ ∀ a b : ℝ, conditionC a b) ∧
  (¬ (solutionSetD = {x : ℝ | 1 / x > 1})) :=
sorry

end NUMINAMATH_CALUDE_conclusions_C_and_D_are_incorrect_l806_80657


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l806_80652

/-- Represents the volume ratio of two tanks given specific oil transfer conditions -/
theorem tank_volume_ratio (tank1 tank2 : ℚ) : 
  tank1 > 0 → 
  tank2 > 0 → 
  (3/4 : ℚ) * tank1 = (2/5 : ℚ) * tank2 → 
  tank1 / tank2 = 8/15 := by
  sorry

#check tank_volume_ratio

end NUMINAMATH_CALUDE_tank_volume_ratio_l806_80652


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l806_80681

theorem binomial_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l806_80681


namespace NUMINAMATH_CALUDE_series_sum_l806_80646

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((2 * n - 3) * a - (n - 2) * b) * (2 * n * a - (2 * n - 1) * b))
  ∑' n, series_term n = 1 / ((a - b) * b) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l806_80646


namespace NUMINAMATH_CALUDE_first_discount_percentage_l806_80607

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 32 →
  final_price = 18 →
  second_discount = 0.25 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l806_80607


namespace NUMINAMATH_CALUDE_x_value_l806_80696

theorem x_value (x : ℝ) : x + Real.sqrt 81 = 25 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l806_80696


namespace NUMINAMATH_CALUDE_planned_goats_addition_l806_80667

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm -/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The planned additions to the farm -/
def plannedAdditions : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 0 }

/-- The final desired number of animals -/
def finalTotal : ℕ := 21

/-- Theorem: The number of goats the farmer plans to add is 2 -/
theorem planned_goats_addition :
  finalTotal = totalAnimals initialFarm + totalAnimals plannedAdditions + 2 := by
  sorry

end NUMINAMATH_CALUDE_planned_goats_addition_l806_80667


namespace NUMINAMATH_CALUDE_perpendicular_necessary_and_sufficient_l806_80632

/-- A plane -/
structure Plane where
  dummy : Unit

/-- A line in a plane -/
structure Line (α : Plane) where
  dummy : Unit

/-- Predicate for a line being straight -/
def isStraight (α : Plane) (l : Line α) : Prop :=
  sorry

/-- Predicate for a line being oblique -/
def isOblique (α : Plane) (m : Line α) : Prop :=
  sorry

/-- Predicate for two lines being perpendicular -/
def isPerpendicular (α : Plane) (l m : Line α) : Prop :=
  sorry

/-- Theorem stating that for a straight line l and an oblique line m on plane α,
    l being perpendicular to m is both necessary and sufficient -/
theorem perpendicular_necessary_and_sufficient (α : Plane) (l m : Line α)
    (h1 : isStraight α l) (h2 : isOblique α m) :
    isPerpendicular α l m ↔ True :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_and_sufficient_l806_80632


namespace NUMINAMATH_CALUDE_group_abelian_l806_80697

variable {G : Type*} [Group G]

theorem group_abelian (h : ∀ x : G, x * x = 1) : ∀ a b : G, a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_group_abelian_l806_80697


namespace NUMINAMATH_CALUDE_square_field_area_l806_80622

/-- Given a square field where a horse takes 10 hours to run around it at a speed of 16 km/h, 
    the area of the field is 1600 square kilometers. -/
theorem square_field_area (s : ℝ) : 
  s > 0 → -- s is positive (side length of square)
  (4 * s = 16 * 10) → -- perimeter equals distance traveled by horse
  s^2 = 1600 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l806_80622


namespace NUMINAMATH_CALUDE_equation_solution_l806_80679

theorem equation_solution : ∃ x : ℚ, (x - 2)^2 - (x + 3)*(x - 3) = 4*x - 1 ∧ x = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l806_80679


namespace NUMINAMATH_CALUDE_total_profit_is_4650_l806_80606

/-- Given the capitals of three individuals P, Q, and R, and the profit share of R,
    calculate the total profit. -/
def calculate_total_profit (Cp Cq Cr R_share : ℚ) : ℚ :=
  let total_ratio := (10 : ℚ) / 4 + 10 / 6 + 1
  R_share * total_ratio / (1 : ℚ)

/-- Theorem stating that under given conditions, the total profit is 4650. -/
theorem total_profit_is_4650 (Cp Cq Cr : ℚ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) 
    (h3 : calculate_total_profit Cp Cq Cr 900 = 4650) : 
  calculate_total_profit Cp Cq Cr 900 = 4650 := by
  sorry

#eval calculate_total_profit 1 1 1 900

end NUMINAMATH_CALUDE_total_profit_is_4650_l806_80606


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l806_80691

theorem parametric_to_cartesian :
  ∀ x y θ : ℝ,
  x = Real.sin θ →
  y = Real.cos (2 * θ) →
  -1 ≤ x ∧ x ≤ 1 →
  y = 1 - 2 * x^2 :=
by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l806_80691


namespace NUMINAMATH_CALUDE_double_root_condition_l806_80695

/-- For a polynomial of the form A x^(n+1) + B x^n + 1, where n is a natural number,
    x = 1 is a root with multiplicity at least 2 if and only if A = n and B = -(n+1). -/
theorem double_root_condition (n : ℕ) (A B : ℝ) :
  (∀ x : ℝ, A * x^(n+1) + B * x^n + 1 = 0 ∧ 
   (A * (n+1) * x^n + B * n * x^(n-1) = 0)) ↔ 
  (A = n ∧ B = -(n+1)) :=
sorry

end NUMINAMATH_CALUDE_double_root_condition_l806_80695


namespace NUMINAMATH_CALUDE_solve_system_l806_80690

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l806_80690


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l806_80618

/-- Represents the daily wage in Rupees -/
def daily_wage : ℚ := 25

/-- Represents the daily fine in Rupees -/
def daily_fine : ℚ := 7.5

/-- Represents the total amount received in Rupees -/
def total_amount : ℚ := 620

/-- Represents the number of absent days -/
def absent_days : ℕ := 4

/-- Theorem stating that the number of days the contractor was engaged is 26 -/
theorem contractor_engagement_days : 
  ∃ (work_days : ℕ), 
    (daily_wage * work_days - daily_fine * absent_days = total_amount) ∧ 
    (work_days + absent_days = 26) := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l806_80618
