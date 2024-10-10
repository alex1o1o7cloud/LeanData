import Mathlib

namespace rectangle_perimeter_ratio_l35_3513

theorem rectangle_perimeter_ratio :
  let original_width : ℚ := 6
  let original_height : ℚ := 8
  let folded_height : ℚ := original_height / 2
  let small_width : ℚ := original_width / 2
  let small_height : ℚ := folded_height
  let original_perimeter : ℚ := 2 * (original_width + original_height)
  let small_perimeter : ℚ := 2 * (small_width + small_height)
  small_perimeter / original_perimeter = 1 / 2 := by
  sorry

end rectangle_perimeter_ratio_l35_3513


namespace remainder_theorem_l35_3571

theorem remainder_theorem : ∃ q : ℕ, 2^222 + 222 = q * (2^111 + 2^56 + 1) + 218 := by
  sorry

end remainder_theorem_l35_3571


namespace problem_hexagon_area_l35_3576

/-- Represents a hexagon formed by stretching a rubber band over pegs on a grid. -/
structure Hexagon where
  interior_points : ℕ
  boundary_points : ℕ

/-- Calculates the area of a hexagon using Pick's Theorem. -/
def area (h : Hexagon) : ℕ :=
  h.interior_points + h.boundary_points / 2 - 1

/-- The hexagon formed on the 5x5 grid as described in the problem. -/
def problem_hexagon : Hexagon :=
  { interior_points := 11
  , boundary_points := 6 }

/-- Theorem stating that the area of the problem hexagon is 13 square units. -/
theorem problem_hexagon_area :
  area problem_hexagon = 13 := by
  sorry

#eval area problem_hexagon  -- Should output 13

end problem_hexagon_area_l35_3576


namespace alicia_tax_deduction_l35_3545

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.2%, the local tax deduction is 55 cents. -/
theorem alicia_tax_deduction :
  localTaxDeduction 25 2.2 = 55 := by
  sorry

end alicia_tax_deduction_l35_3545


namespace not_p_necessary_not_sufficient_for_not_p_or_q_l35_3599

theorem not_p_necessary_not_sufficient_for_not_p_or_q :
  (∃ p q : Prop, ¬p ∧ (p ∨ q)) ∧
  (∀ p q : Prop, ¬(p ∨ q) → ¬p) :=
by sorry

end not_p_necessary_not_sufficient_for_not_p_or_q_l35_3599


namespace least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l35_3574

theorem least_k_divisible_by_1260 (k : ℕ) : k > 0 ∧ k^4 % 1260 = 0 → k ≥ 210 := by
  sorry

theorem two_ten_divisible_by_1260 : (210 : ℕ)^4 % 1260 = 0 := by
  sorry

theorem least_k_is_210 : ∃ k : ℕ, k > 0 ∧ k^4 % 1260 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m^4 % 1260 = 0) → m ≥ k :=
  ⟨210, by
    sorry⟩

end least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l35_3574


namespace unique_base_l35_3549

/-- Converts a number from base h to base 10 --/
def to_base_10 (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The equation in base h --/
def equation_holds (h : Nat) : Prop :=
  h > 9 ∧ 
  to_base_10 [8, 3, 2, 7] h + to_base_10 [9, 4, 6, 1] h = to_base_10 [1, 9, 2, 8, 8] h

theorem unique_base : ∃! h, equation_holds h :=
  sorry

end unique_base_l35_3549


namespace square_weight_calculation_l35_3525

theorem square_weight_calculation (density : ℝ) (thickness : ℝ) 
  (side_length1 : ℝ) (weight1 : ℝ) (side_length2 : ℝ) 
  (h1 : density > 0) (h2 : thickness > 0) 
  (h3 : side_length1 = 4) (h4 : weight1 = 16) (h5 : side_length2 = 6) :
  let weight2 := density * thickness * side_length2^2
  weight2 = 36 := by
  sorry

#check square_weight_calculation

end square_weight_calculation_l35_3525


namespace sqrt_square_abs_sqrt_neg_five_squared_l35_3517

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by sorry

end sqrt_square_abs_sqrt_neg_five_squared_l35_3517


namespace sqrt_difference_equals_three_l35_3560

theorem sqrt_difference_equals_three : Real.sqrt (81 + 49) - Real.sqrt (36 + 25) = 3 := by
  sorry

end sqrt_difference_equals_three_l35_3560


namespace fence_whitewashing_fence_theorem_l35_3511

theorem fence_whitewashing (total_fence : ℝ) (ben_amount : ℝ) 
  (billy_fraction : ℝ) (johnny_fraction : ℝ) : ℝ :=
  let remaining_after_ben := total_fence - ben_amount
  let billy_amount := billy_fraction * remaining_after_ben
  let remaining_after_billy := remaining_after_ben - billy_amount
  let johnny_amount := johnny_fraction * remaining_after_billy
  let final_remaining := remaining_after_billy - johnny_amount
  final_remaining

theorem fence_theorem : 
  fence_whitewashing 100 10 (1/5) (1/3) = 48 := by
  sorry

end fence_whitewashing_fence_theorem_l35_3511


namespace angle_bisector_length_l35_3532

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (DF : ℝ)

-- Define the angle bisector EG
def angleBisector (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_length (t : Triangle) 
  (h1 : t.DE = 4)
  (h2 : t.EF = 5)
  (h3 : t.DF = 6) :
  angleBisector t = 3 * Real.sqrt 6 := by sorry

end angle_bisector_length_l35_3532


namespace article_cost_price_l35_3597

theorem article_cost_price (C S : ℝ) : 
  (S = 1.05 * C) →                    -- Condition 1
  (S - 5 = 1.1 * (0.95 * C)) →        -- Condition 2
  C = 1000 :=                         -- Conclusion
by sorry

end article_cost_price_l35_3597


namespace slope_of_sine_at_pi_fourth_l35_3559

theorem slope_of_sine_at_pi_fourth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π/4) = Real.sqrt 2 / 2 := by
  sorry

end slope_of_sine_at_pi_fourth_l35_3559


namespace triangle_inradius_circumradius_inequality_l35_3557

/-- For any triangle with inradius r, circumradius R, and an angle α, 
    the inequality r / R ≤ 2 sin(α / 2)(1 - sin(α / 2)) holds. -/
theorem triangle_inradius_circumradius_inequality 
  (r R α : ℝ) 
  (hr : r > 0) 
  (hR : R > 0) 
  (hα : 0 < α ∧ α < π) : 
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end triangle_inradius_circumradius_inequality_l35_3557


namespace square_sum_zero_l35_3583

theorem square_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_cubic_heptic : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 0 := by
  sorry

end square_sum_zero_l35_3583


namespace power_division_rule_l35_3582

theorem power_division_rule (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end power_division_rule_l35_3582


namespace height_to_hypotenuse_l35_3509

theorem height_to_hypotenuse (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → a^2 + b^2 = c^2 → (a * b) / 2 = (c * h) / 2 → h = 4.8 :=
by sorry

end height_to_hypotenuse_l35_3509


namespace changgi_weight_l35_3568

/-- Given the weights of three people with certain relationships, prove Changgi's weight -/
theorem changgi_weight (total_weight chaeyoung_hyeonjeong_diff changgi_chaeyoung_diff : ℝ) 
  (h1 : total_weight = 106.6)
  (h2 : chaeyoung_hyeonjeong_diff = 7.7)
  (h3 : changgi_chaeyoung_diff = 4.8) : 
  ∃ (changgi chaeyoung hyeonjeong : ℝ),
    changgi + chaeyoung + hyeonjeong = total_weight ∧
    chaeyoung = hyeonjeong + chaeyoung_hyeonjeong_diff ∧
    changgi = chaeyoung + changgi_chaeyoung_diff ∧
    changgi = 41.3 := by
  sorry

end changgi_weight_l35_3568


namespace pets_lost_l35_3535

/-- Proves the number of pets Anthony lost when he forgot to lock the door -/
theorem pets_lost (initial_pets : ℕ) (final_pets : ℕ) : 
  initial_pets = 16 → 
  final_pets = 8 → 
  (initial_pets - (initial_pets - (initial_pets - final_pets) * 4 / 5)) = final_pets →
  initial_pets - (initial_pets - final_pets) * 5 / 4 = 6 :=
by
  sorry

#check pets_lost

end pets_lost_l35_3535


namespace smallest_sum_five_consecutive_primes_l35_3595

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if five consecutive natural numbers are all prime -/
def fiveConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3) ∧ isPrime (n + 4)

/-- The sum of five consecutive natural numbers starting from n -/
def sumFiveConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3) + (n + 4)

/-- The main theorem: 119 is the smallest sum of five consecutive primes divisible by 5 -/
theorem smallest_sum_five_consecutive_primes :
  ∃ n : ℕ, fiveConsecutivePrimes n ∧ 
           sumFiveConsecutive n = 119 ∧
           119 % 5 = 0 ∧
           (∀ m : ℕ, m < n → fiveConsecutivePrimes m → sumFiveConsecutive m % 5 ≠ 0) :=
sorry

end smallest_sum_five_consecutive_primes_l35_3595


namespace second_quadrant_complex_number_range_l35_3536

theorem second_quadrant_complex_number_range (m : ℝ) : 
  let z : ℂ := m - 1 + (m + 2) * I
  (z.re < 0 ∧ z.im > 0) ↔ -2 < m ∧ m < 1 := by sorry

end second_quadrant_complex_number_range_l35_3536


namespace left_of_kolya_l35_3501

/-- The number of people in a class lineup -/
structure ClassLineup where
  total : ℕ
  leftOfSasha : ℕ
  rightOfSasha : ℕ
  rightOfKolya : ℕ
  leftOfKolya : ℕ

/-- Theorem stating the number of people to the left of Kolya -/
theorem left_of_kolya (c : ClassLineup)
  (h1 : c.leftOfSasha = 20)
  (h2 : c.rightOfSasha = 8)
  (h3 : c.rightOfKolya = 12)
  (h4 : c.total = c.leftOfSasha + c.rightOfSasha + 1)
  (h5 : c.total = c.leftOfKolya + c.rightOfKolya + 1) :
  c.leftOfKolya = 16 := by
  sorry

end left_of_kolya_l35_3501


namespace max_radius_in_wine_glass_l35_3586

theorem max_radius_in_wine_glass :
  let f : ℝ → ℝ := λ x ↦ x^4
  let max_r : ℝ := (3/4) * Real.rpow 2 (1/3)
  ∀ r > 0,
    (∀ x y : ℝ, (y - r)^2 + x^2 = r^2 → y ≥ f x) ∧
    (0 - r)^2 + 0^2 = r^2 →
    r ≤ max_r :=
by sorry

end max_radius_in_wine_glass_l35_3586


namespace kaiden_first_week_cans_l35_3542

/-- The number of cans collected in the first week of Kaiden's soup can collection -/
def cans_first_week (goal : ℕ) (cans_second_week : ℕ) (cans_needed : ℕ) : ℕ :=
  goal - cans_needed - cans_second_week

/-- Theorem stating that Kaiden collected 158 cans in the first week -/
theorem kaiden_first_week_cans :
  cans_first_week 500 259 83 = 158 := by sorry

end kaiden_first_week_cans_l35_3542


namespace rabbit_position_after_ten_exchanges_l35_3552

-- Define the seats
inductive Seat
| one
| two
| three
| four

-- Define the animals
inductive Animal
| mouse
| monkey
| rabbit
| cat

-- Define the seating arrangement
def Arrangement := Seat → Animal

-- Define the initial arrangement
def initial_arrangement : Arrangement := fun seat =>
  match seat with
  | Seat.one => Animal.mouse
  | Seat.two => Animal.monkey
  | Seat.three => Animal.rabbit
  | Seat.four => Animal.cat

-- Define a single exchange operation
def exchange (arr : Arrangement) (n : ℕ) : Arrangement := 
  if n % 2 = 0 then
    fun seat =>
      match seat with
      | Seat.one => arr Seat.three
      | Seat.two => arr Seat.four
      | Seat.three => arr Seat.one
      | Seat.four => arr Seat.two
  else
    fun seat =>
      match seat with
      | Seat.one => arr Seat.two
      | Seat.two => arr Seat.one
      | Seat.three => arr Seat.four
      | Seat.four => arr Seat.three

-- Define multiple exchanges
def multiple_exchanges (arr : Arrangement) (n : ℕ) : Arrangement :=
  match n with
  | 0 => arr
  | n+1 => exchange (multiple_exchanges arr n) n

-- Theorem statement
theorem rabbit_position_after_ten_exchanges :
  ∃ (seat : Seat), (multiple_exchanges initial_arrangement 10) seat = Animal.rabbit ∧ seat = Seat.two :=
sorry

end rabbit_position_after_ten_exchanges_l35_3552


namespace g_composition_l35_3514

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition : g (g (g 3)) = 107 := by
  sorry

end g_composition_l35_3514


namespace smallest_k_multiple_of_200_l35_3556

theorem smallest_k_multiple_of_200 : ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → n < k → ¬(200 ∣ (n * (n + 1) * (2 * n + 1)) / 6)) ∧ 
  (200 ∣ (k * (k + 1) * (2 * k + 1)) / 6) ∧
  k = 31 := by
sorry

end smallest_k_multiple_of_200_l35_3556


namespace thirteenth_term_of_arithmetic_sequence_l35_3566

/-- An arithmetic sequence is defined by its third and twenty-third terms -/
def arithmetic_sequence (a₃ a₂₃ : ℚ) :=
  ∃ (a : ℕ → ℚ), (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ a 3 = a₃ ∧ a 23 = a₂₃

/-- The thirteenth term of the sequence is the average of the third and twenty-third terms -/
theorem thirteenth_term_of_arithmetic_sequence 
  (h : arithmetic_sequence (2/11) (3/7)) : 
  ∃ (a : ℕ → ℚ), a 13 = 47/154 := by
  sorry

end thirteenth_term_of_arithmetic_sequence_l35_3566


namespace angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l35_3562

/-- The angle between clock hands at 9:15 --/
theorem angle_between_clock_hands_at_9_15 : ℝ :=
  let full_rotation : ℝ := 360
  let hours_on_clock_face : ℕ := 12
  let minutes_on_clock_face : ℕ := 60
  let current_hour : ℕ := 9
  let current_minute : ℕ := 15

  let angle_per_hour : ℝ := full_rotation / hours_on_clock_face
  let angle_per_minute : ℝ := full_rotation / minutes_on_clock_face

  let minute_hand_angle : ℝ := current_minute * angle_per_minute
  let hour_hand_angle : ℝ := current_hour * angle_per_hour + (current_minute * angle_per_hour / minutes_on_clock_face)

  let angle_between_hands : ℝ := abs (minute_hand_angle - hour_hand_angle)

  82.5

theorem angle_between_clock_hands_at_9_15_is_82_point_5 :
  angle_between_clock_hands_at_9_15 = 82.5 := by sorry

end angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l35_3562


namespace sphere_surface_area_l35_3594

/-- A rectangular parallelepiped with edge lengths 1, 2, and 3 -/
structure Parallelepiped where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge1_eq : edge1 = 1
  edge2_eq : edge2 = 2
  edge3_eq : edge3 = 3

/-- A sphere containing all vertices of a rectangular parallelepiped -/
structure Sphere where
  radius : ℝ
  contains_parallelepiped : Parallelepiped → Prop

/-- The surface area of a sphere is 14π given the conditions -/
theorem sphere_surface_area (s : Sphere) (p : Parallelepiped) 
  (h : s.contains_parallelepiped p) : 
  s.radius^2 * (4 * Real.pi) = 14 * Real.pi := by
  sorry

end sphere_surface_area_l35_3594


namespace jims_age_l35_3524

theorem jims_age (j t : ℕ) (h1 : j = 3 * t + 10) (h2 : j + t = 70) : j = 55 := by
  sorry

end jims_age_l35_3524


namespace mystery_book_shelves_l35_3553

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 9 →
  picture_book_shelves = 2 →
  total_books = 72 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 :=
by sorry

end mystery_book_shelves_l35_3553


namespace horner_rule_f_at_2_f_2_equals_62_l35_3534

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_f_at_2 :
  horner_eval [2, 3, 0, 5, -4] 2 = f 2 := by sorry

theorem f_2_equals_62 : f 2 = 62 := by sorry

end horner_rule_f_at_2_f_2_equals_62_l35_3534


namespace complex_roots_on_circle_l35_3558

theorem complex_roots_on_circle : 
  ∃ (r : ℝ), r = 2/3 ∧ 
  ∀ (z : ℂ), (z + 1)^5 = 32 * z^5 → Complex.abs z = r :=
sorry

end complex_roots_on_circle_l35_3558


namespace distance_minimized_at_eight_sevenths_l35_3504

/-- Given two points A and B in 3D space, prove that their distance is minimized when x = 8/7 -/
theorem distance_minimized_at_eight_sevenths (x : ℝ) :
  let A := (x, 5 - x, 2*x - 1)
  let B := (1, x + 2, 2 - x)
  let distance := Real.sqrt ((x - 1)^2 + (x + 2 - (5 - x))^2 + (2 - x - (2*x - 1))^2)
  (∀ y : ℝ, distance ≤ Real.sqrt ((y - 1)^2 + (y + 2 - (5 - y))^2 + (2 - y - (2*y - 1))^2)) ↔
  x = 8/7 := by
sorry


end distance_minimized_at_eight_sevenths_l35_3504


namespace calculation_proof_equation_solution_proof_l35_3580

-- Problem 1
theorem calculation_proof :
  18 + |-(Real.sqrt 2)| - (2012 - Real.pi)^0 - 4 * Real.sin (45 * π / 180) = 17 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem equation_solution_proof :
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x) / (x^2 - 4) - 2 / (x - 2) = 1 ∧ x = 0 := by
  sorry

end calculation_proof_equation_solution_proof_l35_3580


namespace roots_equation_value_l35_3591

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^2 + α * (β^2 - 2) = 0 := by
  sorry

end roots_equation_value_l35_3591


namespace greatest_power_of_three_l35_3508

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : (3^k ∣ v) ↔ k ≤ 14 :=
sorry

end greatest_power_of_three_l35_3508


namespace solve_equation_l35_3507

/-- Define the determinant-like operation for four rational numbers -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem stating that given the condition, x must equal 3 -/
theorem solve_equation : ∃ x : ℚ, det (2*x) (-4) x 1 = 18 ∧ x = 3 := by
  sorry

end solve_equation_l35_3507


namespace fixed_point_difference_l35_3596

/-- Given a function f(x) = a^(2x-6) + n, where a > 0 and a ≠ 1,
    and f(m) = 2, prove that m - n = 2 -/
theorem fixed_point_difference (a n m : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  (fun x ↦ a^(2*x - 6) + n) m = 2 → m - n = 2 := by
  sorry

end fixed_point_difference_l35_3596


namespace martha_lasagna_cheese_amount_l35_3585

/-- The amount of cheese Martha needs for her lasagna -/
def cheese_amount : ℝ :=
  1.5

/-- The cost of cheese per kilogram in dollars -/
def cheese_cost_per_kg : ℝ :=
  6

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ :=
  8

/-- The amount of meat Martha needs in grams -/
def meat_amount_grams : ℝ :=
  500

/-- The total cost of ingredients in dollars -/
def total_cost : ℝ :=
  13

theorem martha_lasagna_cheese_amount :
  cheese_amount * cheese_cost_per_kg +
  (meat_amount_grams / 1000) * meat_cost_per_kg =
  total_cost :=
by sorry

end martha_lasagna_cheese_amount_l35_3585


namespace solve_system_of_equations_l35_3572

theorem solve_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 2 * y = 11 ∧ x + 3 * y = 12 ∧ x = 57 / 11 ∧ y = 25 / 11 := by
  sorry

end solve_system_of_equations_l35_3572


namespace spells_conversion_l35_3575

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in base 9 -/
def spellsBase9 : List Nat := [7, 4, 5]

theorem spells_conversion :
  base9ToBase10 spellsBase9 = 448 := by
  sorry

end spells_conversion_l35_3575


namespace least_subtraction_for_divisibility_l35_3592

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 13 ∧ (427398 - x) % 14 = 0 ∧ ∀ y : ℕ, y < x → (427398 - y) % 14 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l35_3592


namespace walter_school_allocation_l35_3563

/-- Represents Walter's work and allocation details -/
structure WorkDetails where
  daysPerWeek : ℕ
  hourlyWage : ℚ
  hoursPerDay : ℕ
  allocationRatio : ℚ

/-- Calculates the amount allocated for school based on work details -/
def schoolAllocation (w : WorkDetails) : ℚ :=
  w.daysPerWeek * w.hourlyWage * w.hoursPerDay * w.allocationRatio

/-- Theorem stating that Walter's school allocation is $75 -/
theorem walter_school_allocation :
  let w : WorkDetails := {
    daysPerWeek := 5,
    hourlyWage := 5,
    hoursPerDay := 4,
    allocationRatio := 3/4
  }
  schoolAllocation w = 75 := by sorry

end walter_school_allocation_l35_3563


namespace slope_theorem_l35_3528

/-- Given two points A(-3,5) and B(x,2) in a coordinate plane, 
    if the slope of the line through A and B is -1/4, then x = 9. -/
theorem slope_theorem (x : ℝ) : 
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (x, 2)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -1/4 → x = 9 := by
sorry

end slope_theorem_l35_3528


namespace three_numbers_in_unit_interval_l35_3577

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) := by
  sorry

end three_numbers_in_unit_interval_l35_3577


namespace incorrect_expression_l35_3531

theorem incorrect_expression (x y : ℝ) (h : x / y = 5 / 6) : 
  ¬((2 * x - y) / y = 4 / 3) := by
sorry

end incorrect_expression_l35_3531


namespace election_result_l35_3505

theorem election_result (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 7500 →
  invalid_percent = 20 / 100 →
  second_candidate_votes = 2700 →
  (↑((total_votes * (1 - invalid_percent)).floor - second_candidate_votes) / ↑((total_votes * (1 - invalid_percent)).floor) : ℚ) = 55 / 100 := by
  sorry

end election_result_l35_3505


namespace sewage_treatment_equipment_costs_l35_3518

theorem sewage_treatment_equipment_costs (a b : ℝ) : 
  (a - b = 3) → (3 * b - 2 * a = 3) → (a = 12 ∧ b = 9) :=
by sorry

end sewage_treatment_equipment_costs_l35_3518


namespace perfect_squares_between_50_and_200_l35_3506

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n : ℕ => 50 < n^2 ∧ n^2 ≤ 200) (Finset.range 201)).card = 7 := by
  sorry

end perfect_squares_between_50_and_200_l35_3506


namespace tv_purchase_price_l35_3515

/-- Proves that the purchase price of a TV is 1200 yuan given the markup, promotion, and profit conditions. -/
theorem tv_purchase_price (x : ℝ) 
  (markup : ℝ → ℝ) 
  (promotion : ℝ → ℝ) 
  (profit : ℝ) 
  (h1 : markup x = 1.35 * x) 
  (h2 : promotion (markup x) = 0.9 * markup x - 50)
  (h3 : promotion (markup x) - x = profit)
  (h4 : profit = 208) : 
  x = 1200 := by
  sorry

end tv_purchase_price_l35_3515


namespace polar_to_cartesian_circle_l35_3550

/-- Prove that the polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x - 2)² + y² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end polar_to_cartesian_circle_l35_3550


namespace expression_bounds_l35_3519

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
    Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ∧
  Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
  Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ≤ 4 * Real.sqrt 2 :=
by sorry


end expression_bounds_l35_3519


namespace power_sum_equals_1999_l35_3589

theorem power_sum_equals_1999 :
  ∃ (a b c d : ℕ), 5^a + 6^b + 7^c + 11^d = 1999 ∧ a = 4 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by
  sorry

end power_sum_equals_1999_l35_3589


namespace arithmetic_sequence_difference_l35_3555

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  first : ℝ  -- First term of the sequence
  d : ℝ       -- Common difference
  seq_def : ∀ n, a n = first + (n - 1) * d
  sum_def : ∀ n, S n = n * (2 * first + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence)
  (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 2) :
  seq.d = 4 := by
  sorry

end arithmetic_sequence_difference_l35_3555


namespace subset_condition_l35_3590

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - a^2 < 0}

theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a < -1 - sqrt 5 ∨ a > (1 + sqrt 5) / 2 := by
  sorry

end subset_condition_l35_3590


namespace benny_missed_games_l35_3573

theorem benny_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : attended_games = 14) :
  total_games - attended_games = 25 := by
  sorry

end benny_missed_games_l35_3573


namespace largest_power_of_ten_in_factorial_l35_3598

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def count_multiples (n : ℕ) (d : ℕ) : ℕ := n / d

def count_factors_of_five (n : ℕ) : ℕ :=
  (count_multiples n 5) + (count_multiples n 25) + (count_multiples n 125)

theorem largest_power_of_ten_in_factorial :
  (∀ k : ℕ, k ≤ 41 → (factorial 170) % (10^k) = 0) ∧
  ¬((factorial 170) % (10^42) = 0) := by
  sorry

end largest_power_of_ten_in_factorial_l35_3598


namespace base_five_digits_of_1234_l35_3510

theorem base_five_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1234 ∧ 1234 < 5^n :=
  sorry

end base_five_digits_of_1234_l35_3510


namespace cube_root_8000_simplification_l35_3516

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * ((b : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * ((d : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) → d ≥ b) ∧
  a = 20 ∧ b = 1 := by
  sorry

end cube_root_8000_simplification_l35_3516


namespace simplify_and_evaluate_l35_3533

theorem simplify_and_evaluate :
  (∀ x y : ℚ, x = -2 ∧ y = -3 → 6*x - 5*y + 3*y - 2*x = -2) ∧
  (∀ a : ℚ, a = -1/2 → 1/4*(-4*a^2 + 2*a - 8) - (1/2*a - 2) = -1/4) := by
  sorry

end simplify_and_evaluate_l35_3533


namespace corresponding_angles_are_equal_l35_3551

-- Define the concept of angles
def Angle : Type := sorry

-- Define the property of being corresponding angles
def are_corresponding (a b : Angle) : Prop := sorry

-- Theorem statement
theorem corresponding_angles_are_equal (a b : Angle) : 
  are_corresponding a b → a = b := by
  sorry

end corresponding_angles_are_equal_l35_3551


namespace a_not_zero_l35_3581

theorem a_not_zero 
  (a b c d : ℝ) 
  (h1 : a / b < -3 * c / d) 
  (h2 : b * d ≠ 0) 
  (h3 : c = 2 * a) : 
  a ≠ 0 := by
sorry

end a_not_zero_l35_3581


namespace phillips_remaining_money_l35_3548

theorem phillips_remaining_money
  (initial_amount : ℕ)
  (spent_oranges : ℕ)
  (spent_apples : ℕ)
  (spent_candy : ℕ)
  (h1 : initial_amount = 95)
  (h2 : spent_oranges = 14)
  (h3 : spent_apples = 25)
  (h4 : spent_candy = 6) :
  initial_amount - (spent_oranges + spent_apples + spent_candy) = 50 :=
by
  sorry

end phillips_remaining_money_l35_3548


namespace base_conversion_sum_l35_3527

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def C : Nat := 12
def D : Nat := 13

theorem base_conversion_sum :
  let base_8_num := base_to_decimal [5, 3, 7] 8
  let base_14_num := base_to_decimal [5, C, D] 14
  base_8_num + base_14_num = 1512 := by
sorry

end base_conversion_sum_l35_3527


namespace quadratic_inequality_range_l35_3587

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, x^2 - 2*x + 5 - m < 0) ↔ m > 5 := by
  sorry

end quadratic_inequality_range_l35_3587


namespace shortest_diagonal_probability_l35_3503

/-- The number of sides in the regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n / 2

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 5 / 44 := by sorry

end shortest_diagonal_probability_l35_3503


namespace trendy_haircut_cost_is_8_l35_3569

/-- The cost of a trendy haircut -/
def trendy_haircut_cost : ℕ → Prop
| cost => 
  let normal_cost : ℕ := 5
  let special_cost : ℕ := 6
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let total_weekly_earnings : ℕ := 413
  (normal_cost * normal_per_day + special_cost * special_per_day + cost * trendy_per_day) * days_per_week = total_weekly_earnings

theorem trendy_haircut_cost_is_8 : trendy_haircut_cost 8 := by
  sorry

end trendy_haircut_cost_is_8_l35_3569


namespace min_value_inequality_l35_3588

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
by sorry

end min_value_inequality_l35_3588


namespace unique_integer_solution_l35_3541

theorem unique_integer_solution : 
  ∃! x : ℤ, (12*x - 1) * (6*x - 1) * (4*x - 1) * (3*x - 1) = 330 :=
by sorry

end unique_integer_solution_l35_3541


namespace triangle_sine_inequality_l35_3579

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := by
  sorry

end triangle_sine_inequality_l35_3579


namespace algebraic_expression_equality_l35_3520

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y + 8 = 18) :
  3*x - 6*y + 4 = 34 := by
  sorry

end algebraic_expression_equality_l35_3520


namespace sufficient_not_necessary_l35_3544

/-- An arithmetic sequence with first term a₁ > 0 and common ratio q -/
structure ArithmeticSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- Sum of first n terms of an arithmetic sequence -/
def S (as : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Statement: q > 1 is sufficient but not necessary for S₃ + S₅ > 2S₄ -/
theorem sufficient_not_necessary (as : ArithmeticSequence) :
  (∀ as, as.q > 1 → S as 3 + S as 5 > 2 * S as 4) ∧
  ¬(∀ as, S as 3 + S as 5 > 2 * S as 4 → as.q > 1) :=
sorry

end sufficient_not_necessary_l35_3544


namespace twenty_four_bananas_cost_l35_3500

/-- The cost of fruits at Lisa's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 8 apples = 5 oranges

/-- Calculate the number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  let apples := (num_bananas : ℚ) * cost.banana_apple_ratio
  apples * cost.apple_orange_ratio

/-- Theorem: 24 bananas cost approximately as much as 11 oranges -/
theorem twenty_four_bananas_cost (cost : FruitCost) 
  (h1 : cost.banana_apple_ratio = 3 / 4)
  (h2 : cost.apple_orange_ratio = 5 / 8) :
  ⌊bananas_to_oranges cost 24⌋ = 11 := by
  sorry

#eval ⌊(24 : ℚ) * (3 / 4) * (5 / 8)⌋  -- Expected output: 11

end twenty_four_bananas_cost_l35_3500


namespace strictly_decreasing_exponential_range_l35_3526

theorem strictly_decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → a ∈ Set.Ioo (1/2) 1 :=
by sorry

end strictly_decreasing_exponential_range_l35_3526


namespace cosine_rationality_l35_3547

theorem cosine_rationality (k : ℤ) (θ : ℝ) 
  (h1 : k ≥ 3)
  (h2 : ∃ q₁ : ℚ, (↑q₁ : ℝ) = Real.cos ((k - 1) * θ))
  (h3 : ∃ q₂ : ℚ, (↑q₂ : ℝ) = Real.cos (k * θ)) :
  ∃ (n : ℕ), n > k ∧ 
    (∃ q₃ : ℚ, (↑q₃ : ℝ) = Real.cos ((n - 1) * θ)) ∧ 
    (∃ q₄ : ℚ, (↑q₄ : ℝ) = Real.cos (n * θ)) := by
  sorry

end cosine_rationality_l35_3547


namespace price_reduction_percentage_l35_3529

/-- Given a price reduction scenario, prove that the first reduction percentage is 25% -/
theorem price_reduction_percentage (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 60 / 100) = P * (1 - 70 / 100) → x = 25 := by
  sorry

#check price_reduction_percentage

end price_reduction_percentage_l35_3529


namespace arithmetic_sequence_sum_l35_3584

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ+) :
  (a₁ ≤ aₙ) →
  (aₙ = a₁ + (n - 1) * d) →
  3 * (n : ℕ) * (a₁ + aₙ) / 2 = 3774 →
  3 * (Finset.sum (Finset.range n) (λ i => a₁ + i * d)) = 3774 := by
  sorry

#check arithmetic_sequence_sum 50 98 3 17

end arithmetic_sequence_sum_l35_3584


namespace total_sleep_time_in_week_l35_3554

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for both a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep * days_in_week) + ((cougar_sleep + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry


end total_sleep_time_in_week_l35_3554


namespace quadratic_inequality_solution_sets_l35_3546

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let f := fun x => x^2 - (1 + a) * x + a
  (a = 2 → {x | f x > 0} = {x | x > 2 ∨ x < 1}) ∧
  (a > 1 → {x | f x > 0} = {x | x > a ∨ x < 1}) ∧
  (a = 1 → {x | f x > 0} = {x | x ≠ 1}) ∧
  (a < 1 → {x | f x > 0} = {x | x > 1 ∨ x < a}) :=
by sorry

end quadratic_inequality_solution_sets_l35_3546


namespace annulus_area_l35_3502

/-- The area of an annulus with outer radius 8 feet and inner radius 2 feet is 60π square feet. -/
theorem annulus_area : ∀ (π : ℝ), π > 0 → π * (8^2 - 2^2) = 60 * π := by
  sorry

end annulus_area_l35_3502


namespace product_remainder_remainder_1287_1499_300_l35_3523

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_1287_1499_300 : (1287 * 1499) % 300 = 213 := by sorry

end product_remainder_remainder_1287_1499_300_l35_3523


namespace third_month_sale_l35_3578

def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = num_months * average_sale - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end third_month_sale_l35_3578


namespace expression_factorization_l35_3530

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 + 49 * x^4 - 9) - (4 * x^6 - 14 * x^4 - 9) = 3 * x^4 * (4 * x^2 + 21) := by
  sorry

end expression_factorization_l35_3530


namespace inequality_and_constraint_solution_l35_3593

-- Define the inequality and its solution set
def inequality (a : ℝ) (x : ℝ) : Prop := 2 * a * x^2 - 8 * x - 3 * a^2 < 0
def solution_set (a b : ℝ) : Set ℝ := {x | -1 < x ∧ x < b ∧ inequality a x}

-- Define the constraint equation
def constraint (a b x y : ℝ) : Prop := a / x + b / y = 1

-- State the theorem
theorem inequality_and_constraint_solution :
  ∃ (a b : ℝ),
    (∀ x, x ∈ solution_set a b ↔ inequality a x) ∧
    a > 0 ∧
    (∀ x y, x > 0 → y > 0 → constraint a b x y →
      3 * x + 2 * y ≥ 24 ∧
      (∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ constraint a b x₀ y₀ ∧ 3 * x₀ + 2 * y₀ = 24)) ∧
    a = 2 ∧
    b = 3 :=
  sorry

end inequality_and_constraint_solution_l35_3593


namespace curve_C_equation_min_area_QAB_l35_3543

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N on parabola E
def point_N (x y : ℝ) : Prop := parabola_E x y

-- Define point O as the origin
def point_O : ℝ × ℝ := (0, 0)

-- Define point P as the midpoint of ON
def point_P (x y : ℝ) : Prop := ∃ (nx ny : ℝ), point_N nx ny ∧ x = nx / 2 ∧ y = ny / 2

-- Define curve C as the trajectory of point P
def curve_C (x y : ℝ) : Prop := point_P x y

-- Define point Q on curve C with x₀ ≥ 5
def point_Q (x₀ y₀ : ℝ) : Prop := curve_C x₀ y₀ ∧ x₀ ≥ 5

-- Theorem for the equation of curve C
theorem curve_C_equation (x y : ℝ) : curve_C x y → y^2 = 4 * x := by sorry

-- Theorem for the minimum area of △QAB
theorem min_area_QAB (x₀ y₀ : ℝ) (hQ : point_Q x₀ y₀) : 
  ∃ (A B : ℝ × ℝ), (∀ (area : ℝ), area ≥ 25/2) := by sorry

end curve_C_equation_min_area_QAB_l35_3543


namespace smaller_circle_circumference_l35_3540

theorem smaller_circle_circumference 
  (R : ℝ) 
  (h1 : R > 0) 
  (h2 : π * (3*R)^2 - π * R^2 = 32 / π) : 
  2 * π * R = 4 := by
sorry

end smaller_circle_circumference_l35_3540


namespace license_plate_combinations_l35_3539

/-- The number of possible letter combinations in the license plate -/
def letter_combinations : ℕ := Nat.choose 26 2 * 3

/-- The number of possible digit combinations in the license plate -/
def digit_combinations : ℕ := 10 * 9 * 3

/-- The total number of possible license plate combinations -/
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem license_plate_combinations :
  total_combinations = 877500 := by
  sorry

end license_plate_combinations_l35_3539


namespace triangle_longest_side_l35_3512

theorem triangle_longest_side (x : ℚ) :
  9 + (x + 5) + (2 * x + 2) = 42 →
  max 9 (max (x + 5) (2 * x + 2)) = 58 / 3 :=
by sorry

end triangle_longest_side_l35_3512


namespace notebook_distribution_l35_3522

theorem notebook_distribution (total notebooks k v y s se : ℕ) : 
  notebooks = 100 ∧
  k + v = 52 ∧
  v + y = 43 ∧
  y + s = 34 ∧
  s + se = 30 ∧
  k + v + y + s + se = notebooks →
  k = 27 ∧ v = 25 ∧ y = 18 ∧ s = 16 ∧ se = 14 := by
  sorry

end notebook_distribution_l35_3522


namespace wine_price_increase_l35_3537

/-- The additional cost for 5 bottles of wine after a 25% price increase -/
theorem wine_price_increase (current_price : ℝ) (num_bottles : ℕ) (price_increase_percent : ℝ) :
  current_price = 20 →
  num_bottles = 5 →
  price_increase_percent = 0.25 →
  num_bottles * current_price * price_increase_percent = 25 :=
by sorry

end wine_price_increase_l35_3537


namespace max_right_triangle_area_in_rectangle_l35_3567

theorem max_right_triangle_area_in_rectangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  ∃ (area : ℝ), area = 90 ∧ 
  ∀ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ a ∧ 
    0 ≤ y ∧ y ≤ b ∧ 
    x^2 + y^2 = z^2 ∧ 
    z ≤ (a^2 + b^2)^(1/2) →
    (1/2) * x * y ≤ area :=
sorry

end max_right_triangle_area_in_rectangle_l35_3567


namespace intersection_sum_l35_3564

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x + 3
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by sorry

end intersection_sum_l35_3564


namespace y_axis_reflection_of_P_l35_3561

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem y_axis_reflection_of_P :
  let P : ℝ × ℝ := (-1, 2)
  reflect_y_axis P = (1, 2) := by sorry

end y_axis_reflection_of_P_l35_3561


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l35_3538

/-- The equation represents a hyperbola if both coefficients are nonzero and have opposite signs -/
def is_hyperbola (k : ℝ) : Prop :=
  k - 3 > 0 ∧ k > 0

/-- k > 3 is a sufficient condition for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k :=
sorry

/-- k > 3 is not a necessary condition for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 3) :=
sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary (k : ℝ) : 
  (k > 3 → is_hyperbola k) ∧ ¬(is_hyperbola k → k > 3) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l35_3538


namespace sqrt_equation_solution_l35_3521

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 5 * z) = 7 :=
by
  -- The unique solution is z = -44/5
  use -44/5
  constructor
  -- Prove that -44/5 satisfies the equation
  · sorry
  -- Prove uniqueness
  · sorry

end sqrt_equation_solution_l35_3521


namespace interest_difference_l35_3570

/-- Calculate the difference between the principal and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 9200 ∧ rate = 12 ∧ time = 3 →
  principal - (principal * rate * time / 100) = 5888 := by
sorry

end interest_difference_l35_3570


namespace coffee_fraction_is_37_84_l35_3565

-- Define the initial conditions
def initial_coffee : ℚ := 5
def initial_cream : ℚ := 7
def cup_size : ℚ := 10

-- Define the transfers
def first_transfer : ℚ := 2
def second_transfer : ℚ := 3
def third_transfer : ℚ := 1

-- Define the function to calculate the final fraction of coffee in cup 1
def final_coffee_fraction (ic : ℚ) (icr : ℚ) (cs : ℚ) (ft : ℚ) (st : ℚ) (tt : ℚ) : ℚ :=
  let coffee_after_first := ic - ft
  let total_after_first := coffee_after_first + icr + ft
  let coffee_ratio_second := ft / total_after_first
  let coffee_returned := st * coffee_ratio_second
  let total_after_second := coffee_after_first + coffee_returned + st * (1 - coffee_ratio_second)
  let coffee_after_second := coffee_after_first + coffee_returned
  let coffee_ratio_third := coffee_after_second / total_after_second
  let coffee_final := coffee_after_second - tt * coffee_ratio_third
  let total_final := total_after_second - tt
  coffee_final / total_final

-- Theorem statement
theorem coffee_fraction_is_37_84 :
  final_coffee_fraction initial_coffee initial_cream cup_size first_transfer second_transfer third_transfer = 37 / 84 := by
  sorry

end coffee_fraction_is_37_84_l35_3565
