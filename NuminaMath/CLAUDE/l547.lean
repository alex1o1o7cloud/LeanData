import Mathlib

namespace NUMINAMATH_CALUDE_heart_ratio_equals_one_l547_54794

def heart (n m : ℕ) : ℕ := n^3 * m^3

theorem heart_ratio_equals_one : (heart 3 5) / (heart 5 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_equals_one_l547_54794


namespace NUMINAMATH_CALUDE_multiple_of_119_l547_54725

theorem multiple_of_119 : ∃ k : ℤ, 119 = 7 * k ∧ 
  (∀ m : ℤ, 119 ≠ 2 * m) ∧ 
  (∀ n : ℤ, 119 ≠ 3 * n) ∧ 
  (∀ p : ℤ, 119 ≠ 5 * p) ∧ 
  (∀ q : ℤ, 119 ≠ 11 * q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_119_l547_54725


namespace NUMINAMATH_CALUDE_product_increase_2016_l547_54778

theorem product_increase_2016 : ∃ (a b c : ℕ), 
  ((a - 3) * (b - 3) * (c - 3)) - (a * b * c) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_2016_l547_54778


namespace NUMINAMATH_CALUDE_sum_of_fraction_is_correct_l547_54717

/-- The repeating decimal 0.̅14 as a real number -/
def repeating_decimal : ℚ := 14 / 99

/-- The sum of numerator and denominator of the fraction representation of 0.̅14 -/
def sum_of_fraction : ℕ := 113

/-- Theorem stating that the sum of numerator and denominator of 0.̅14 in lowest terms is 113 -/
theorem sum_of_fraction_is_correct : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_is_correct_l547_54717


namespace NUMINAMATH_CALUDE_parallelogram_area_l547_54718

/-- The area of a parallelogram with given side lengths and angle between them -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 15) (hb : b = 20) (hθ : θ = 35 * π / 180) :
  abs (a * b * Real.sin θ - 172.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l547_54718


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l547_54723

theorem complex_magnitude_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (z - Complex.abs z = 4 - 6*I) → Complex.normSq z = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l547_54723


namespace NUMINAMATH_CALUDE_quadratic_factorization_l547_54786

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l547_54786


namespace NUMINAMATH_CALUDE_hyperbola_condition_l547_54702

theorem hyperbola_condition (m : ℝ) :
  (m > 0 → m * (m + 2) > 0) ∧ ¬(m * (m + 2) > 0 → m > 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l547_54702


namespace NUMINAMATH_CALUDE_football_count_proof_l547_54705

/-- The cost of a single soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The cost of some footballs and 3 soccer balls in dollars -/
def first_set_cost : ℕ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℕ := 155

/-- The number of footballs in the second set -/
def footballs_in_second_set : ℕ := 3

theorem football_count_proof : 
  ∃ (football_cost : ℕ) (footballs_in_first_set : ℕ),
    footballs_in_first_set * football_cost + 3 * soccer_ball_cost = first_set_cost ∧
    3 * football_cost + soccer_ball_cost = second_set_cost ∧
    footballs_in_second_set = 3 :=
sorry

end NUMINAMATH_CALUDE_football_count_proof_l547_54705


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l547_54792

theorem infinitely_many_non_representable : 
  ∃ f : ℕ → ℤ, Function.Injective f ∧ 
    ∀ (k : ℕ) (a b c : ℕ), f k ≠ 2^a + 3^b - 5^c := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l547_54792


namespace NUMINAMATH_CALUDE_shaded_perimeter_equals_48_l547_54704

/-- Represents a circle in the arrangement -/
structure Circle where
  circumference : ℝ

/-- Represents the arrangement of four circles -/
structure CircleArrangement where
  circles : Fin 4 → Circle
  symmetric : Bool
  touching : Bool

/-- Calculates the perimeter of the shaded region -/
def shadedPerimeter (arrangement : CircleArrangement) : ℝ :=
  sorry

theorem shaded_perimeter_equals_48 (arrangement : CircleArrangement) 
    (h1 : ∀ i, (arrangement.circles i).circumference = 48) 
    (h2 : arrangement.symmetric = true) 
    (h3 : arrangement.touching = true) : 
  shadedPerimeter arrangement = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_perimeter_equals_48_l547_54704


namespace NUMINAMATH_CALUDE_sci_fi_readers_l547_54796

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 250 → literary = 88 → both = 18 → sci_fi = total + both - literary :=
by
  sorry

end NUMINAMATH_CALUDE_sci_fi_readers_l547_54796


namespace NUMINAMATH_CALUDE_min_gumballs_for_three_same_color_l547_54771

/-- Represents the colors of gumballs in the machine -/
inductive GumballColor
| Red
| Blue
| White
| Green

/-- Represents the gumball machine -/
structure GumballMachine where
  red : Nat
  blue : Nat
  white : Nat
  green : Nat

/-- Returns the minimum number of gumballs needed to guarantee 3 of the same color -/
def minGumballsForThreeSameColor (machine : GumballMachine) : Nat :=
  sorry

/-- Theorem stating that for the given gumball machine, 
    the minimum number of gumballs needed to guarantee 3 of the same color is 8 -/
theorem min_gumballs_for_three_same_color :
  let machine : GumballMachine := { red := 13, blue := 5, white := 1, green := 9 }
  minGumballsForThreeSameColor machine = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_three_same_color_l547_54771


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_800_l547_54738

theorem modular_inverse_of_7_mod_800 :
  let a : ℕ := 7
  let m : ℕ := 800
  let inv : ℕ := 343
  (Nat.gcd a m = 1) →
  (inv < m) →
  (a * inv) % m = 1 →
  ∃ x : ℕ, x < m ∧ (a * x) % m = 1 ∧ x = inv :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_800_l547_54738


namespace NUMINAMATH_CALUDE_jake_has_seven_balls_l547_54750

/-- The number of balls Audrey has -/
def audrey_balls : ℕ := 41

/-- The difference in the number of balls between Audrey and Jake -/
def difference : ℕ := 34

/-- The number of balls Jake has -/
def jake_balls : ℕ := audrey_balls - difference

/-- Theorem stating that Jake has 7 balls -/
theorem jake_has_seven_balls : jake_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_balls_l547_54750


namespace NUMINAMATH_CALUDE_equation_solutions_count_l547_54707

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ θ => 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ)
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    solutions.card = 4 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l547_54707


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l547_54724

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l547_54724


namespace NUMINAMATH_CALUDE_prob_empty_mailbox_is_five_ninths_l547_54784

/-- The number of different greeting cards -/
def num_cards : ℕ := 4

/-- The number of different mailboxes -/
def num_mailboxes : ℕ := 3

/-- The probability of at least one mailbox being empty when cards are randomly placed -/
def prob_empty_mailbox : ℚ := 5/9

/-- Theorem stating that the probability of at least one empty mailbox is 5/9 -/
theorem prob_empty_mailbox_is_five_ninths :
  prob_empty_mailbox = 5/9 :=
sorry

end NUMINAMATH_CALUDE_prob_empty_mailbox_is_five_ninths_l547_54784


namespace NUMINAMATH_CALUDE_sum_of_k_values_l547_54711

theorem sum_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^2 / (1 - b) = k)
  (h_eq2 : b^2 / (1 - c) = k)
  (h_eq3 : c^2 / (1 - a) = k) :
  ∃ k1 k2 : ℝ, k = k1 ∨ k = k2 ∧ k1 + k2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l547_54711


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_isosceles_triangle_4_exists_l547_54753

/-- An isosceles triangle with perimeter 18 and legs twice the base length -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 18
  leg_eq : leg = 2 * base

/-- An isosceles triangle with one side 4 -/
structure IsoscelesTriangle4 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter_eq : side1 + side2 + side3 = 18
  isosceles_eq : side2 = side3
  one_side_4 : side1 = 4 ∨ side2 = 4

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  t.base = 18 / 5 ∧ t.leg = 36 / 5 := by sorry

theorem isosceles_triangle_4_exists :
  ∃ (t : IsoscelesTriangle4), t.side2 = 7 ∧ t.side3 = 7 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_isosceles_triangle_4_exists_l547_54753


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l547_54720

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + 2*I
  second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l547_54720


namespace NUMINAMATH_CALUDE_cube_of_complex_number_l547_54781

theorem cube_of_complex_number :
  let z : ℂ := 2 + 5*I
  z^3 = -142 - 65*I := by sorry

end NUMINAMATH_CALUDE_cube_of_complex_number_l547_54781


namespace NUMINAMATH_CALUDE_roots_are_irrational_l547_54780

theorem roots_are_irrational (j : ℝ) : 
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ x * y = 11) →
  (∃ x y : ℝ, x^2 - 5*j*x + 3*j^2 - 2 = 0 ∧ y^2 - 5*j*y + 3*j^2 - 2 = 0 ∧ ¬(∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l547_54780


namespace NUMINAMATH_CALUDE_complex_root_implies_positive_and_triangle_l547_54745

theorem complex_root_implies_positive_and_triangle (a b c : ℝ) 
  (h_root : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ Complex.I * Complex.I = -1 ∧ 
    (α + Complex.I * β) ^ 2 - (a + b + c) * (α + Complex.I * β) + (a * b + b * c + c * a) = 0) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ (Real.sqrt a < Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_implies_positive_and_triangle_l547_54745


namespace NUMINAMATH_CALUDE_salary_calculation_l547_54741

/-- Represents the man's original monthly salary in Rupees -/
def original_salary : ℝ := sorry

/-- The man's original savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- The man's original rent expense rate as a decimal -/
def rent_rate : ℝ := 0.40

/-- The man's original utilities expense rate as a decimal -/
def utilities_rate : ℝ := 0.30

/-- The man's original groceries expense rate as a decimal -/
def groceries_rate : ℝ := 0.20

/-- The increase rate for rent as a decimal -/
def rent_increase : ℝ := 0.15

/-- The increase rate for utilities as a decimal -/
def utilities_increase : ℝ := 0.20

/-- The increase rate for groceries as a decimal -/
def groceries_increase : ℝ := 0.10

/-- The reduced savings amount in Rupees -/
def reduced_savings : ℝ := 180

theorem salary_calculation :
  original_salary * savings_rate - reduced_savings =
  original_salary * (rent_rate * (1 + rent_increase) +
                     utilities_rate * (1 + utilities_increase) +
                     groceries_rate * (1 + groceries_increase)) -
  original_salary * (rent_rate + utilities_rate + groceries_rate) ∧
  original_salary = 3000 :=
sorry

end NUMINAMATH_CALUDE_salary_calculation_l547_54741


namespace NUMINAMATH_CALUDE_youtube_views_multiple_l547_54722

/-- The multiple by which views increased on the fourth day -/
def viewMultiple (initialViews : ℕ) (totalViews : ℕ) (additionalViews : ℕ) : ℚ :=
  (totalViews - additionalViews - initialViews) / initialViews

theorem youtube_views_multiple :
  let initialViews : ℕ := 4000
  let totalViews : ℕ := 94000
  let additionalViews : ℕ := 50000
  viewMultiple initialViews totalViews additionalViews = 11 := by
sorry

end NUMINAMATH_CALUDE_youtube_views_multiple_l547_54722


namespace NUMINAMATH_CALUDE_lars_bakeshop_production_l547_54716

/-- Lars' bakeshop productivity calculation -/
theorem lars_bakeshop_production :
  let loaves_per_hour : ℕ := 10
  let baguettes_per_two_hours : ℕ := 30
  let working_hours_per_day : ℕ := 6
  
  let loaves_per_day : ℕ := loaves_per_hour * working_hours_per_day
  let baguette_intervals : ℕ := working_hours_per_day / 2
  let baguettes_per_day : ℕ := baguettes_per_two_hours * baguette_intervals
  
  loaves_per_day + baguettes_per_day = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_lars_bakeshop_production_l547_54716


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l547_54700

/-- Given information about students taking geometry and history classes,
    prove that the number of students taking either geometry or history
    but not both is 35. -/
theorem students_taking_one_subject (total_geometry : ℕ)
                                    (both_subjects : ℕ)
                                    (history_only : ℕ)
                                    (h1 : total_geometry = 40)
                                    (h2 : both_subjects = 20)
                                    (h3 : history_only = 15) :
  (total_geometry - both_subjects) + history_only = 35 := by
  sorry


end NUMINAMATH_CALUDE_students_taking_one_subject_l547_54700


namespace NUMINAMATH_CALUDE_min_value_expression_l547_54772

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 ≤ x ∧ x ≤ 1/2) 
  (hy : -1/2 ≤ y ∧ y ≤ 1/2) 
  (hz : -1/2 ≤ z ∧ z ≤ 1/2) : 
  (1/((1 - x^2)*(1 - y^2)*(1 - z^2))) + (1/((1 + x^2)*(1 + y^2)*(1 + z^2))) ≥ 2 ∧
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2))) + (1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l547_54772


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l547_54731

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ a ∈ Set.Icc 1 5 \ {5} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l547_54731


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l547_54756

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

-- Define the number of right angles in the polygon
def right_angles (p : ConvexPolygon) : ℕ := 2

-- Define the function to calculate the number of diagonals
def num_diagonals (p : ConvexPolygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

-- Theorem statement
theorem nine_sided_polygon_diagonals (p : ConvexPolygon) :
  p.sides = 9 → p.is_convex = true → right_angles p = 2 → num_diagonals p = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l547_54756


namespace NUMINAMATH_CALUDE_average_PQR_l547_54777

theorem average_PQR (P Q R : ℚ) 
  (eq1 : 1001 * R - 3003 * P = 6006)
  (eq2 : 2002 * Q + 4004 * P = 8008) :
  (P + Q + R) / 3 = 2 * (P + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_PQR_l547_54777


namespace NUMINAMATH_CALUDE_johns_leftover_earnings_l547_54729

/-- Proves that given John spent 40% of his earnings on rent and 30% less than that on a dishwasher, he had 32% of his earnings left over. -/
theorem johns_leftover_earnings : 
  ∀ (total_earnings : ℝ) (rent_percent : ℝ) (dishwasher_percent : ℝ),
    rent_percent = 40 →
    dishwasher_percent = rent_percent - (0.3 * rent_percent) →
    100 - (rent_percent + dishwasher_percent) = 32 := by
  sorry

end NUMINAMATH_CALUDE_johns_leftover_earnings_l547_54729


namespace NUMINAMATH_CALUDE_egyptian_fractions_unit_sum_l547_54746

theorem egyptian_fractions_unit_sum (a b c : ℕ) : 
  a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 →
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  ((a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) ∨
  ((a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
   (a = 3 ∧ b = 2 ∧ c = 6) ∨ (a = 3 ∧ b = 6 ∧ c = 2) ∨ 
   (a = 6 ∧ b = 2 ∧ c = 3) ∨ (a = 6 ∧ b = 3 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fractions_unit_sum_l547_54746


namespace NUMINAMATH_CALUDE_equal_roots_condition_l547_54754

theorem equal_roots_condition (m : ℝ) : 
  (∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m) →
  (∃! r, ∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m → x = r) ↔ 
  m = 3/2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l547_54754


namespace NUMINAMATH_CALUDE_collect_all_blocks_time_l547_54759

/-- Represents the block collection problem --/
structure BlockCollection where
  totalBlocks : ℕ := 50
  dadPuts : ℕ := 5
  miaRemoves : ℕ := 3
  brotherRemoves : ℕ := 1
  cycleTime : ℕ := 30  -- in seconds

/-- Calculates the time in minutes to collect all blocks --/
def timeToCollectAll (bc : BlockCollection) : ℕ :=
  let netBlocksPerCycle := bc.dadPuts - (bc.miaRemoves + bc.brotherRemoves)
  let cyclesToReachAlmostAll := (bc.totalBlocks - bc.dadPuts) / netBlocksPerCycle
  let totalSeconds := (cyclesToReachAlmostAll + 1) * bc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the time to collect all blocks is 23 minutes --/
theorem collect_all_blocks_time (bc : BlockCollection) :
  timeToCollectAll bc = 23 := by
  sorry

end NUMINAMATH_CALUDE_collect_all_blocks_time_l547_54759


namespace NUMINAMATH_CALUDE_unique_prime_double_squares_l547_54736

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x y : ℕ), p + 7 = 2 * x^2 ∧ p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_double_squares_l547_54736


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l547_54706

theorem factorial_sum_equation (x y : ℕ) (z : ℤ) 
  (h_odd : ∃ k : ℤ, z = 2 * k + 1)
  (h_eq : x.factorial + y.factorial = 48 * z + 2017) :
  ((x = 1 ∧ y = 6 ∧ z = -27) ∨
   (x = 6 ∧ y = 1 ∧ z = -27) ∨
   (x = 1 ∧ y = 7 ∧ z = 63) ∨
   (x = 7 ∧ y = 1 ∧ z = 63)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l547_54706


namespace NUMINAMATH_CALUDE_outfit_choices_l547_54763

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each type of clothing -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfit combinations where shirt and pants are the same color -/
def matching_combinations : ℕ := num_colors * num_items

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - matching_combinations

theorem outfit_choices :
  valid_outfits = 448 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l547_54763


namespace NUMINAMATH_CALUDE_valid_arrangements_eq_48_l547_54768

/-- The number of people in the lineup -/
def n : ℕ := 5

/-- A function that calculates the number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of valid arrangements for 5 people is 48 -/
theorem valid_arrangements_eq_48 : validArrangements n = 48 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_eq_48_l547_54768


namespace NUMINAMATH_CALUDE_income_data_mean_difference_l547_54701

/-- The difference between the mean of incorrect data and the mean of actual data -/
theorem income_data_mean_difference (T : ℝ) : 
  (T + 1200000) / 500 - (T + 120000) / 500 = 2160 := by sorry

end NUMINAMATH_CALUDE_income_data_mean_difference_l547_54701


namespace NUMINAMATH_CALUDE_existence_of_solution_l547_54712

theorem existence_of_solution : ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l547_54712


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l547_54785

theorem jason_pokemon_cards (initial_cards : ℕ) (given_away : ℕ) : 
  initial_cards = 9 → given_away = 4 → initial_cards - given_away = 5 := by
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l547_54785


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_fraction_l547_54779

theorem sum_real_imag_parts_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  (z.re + z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_fraction_l547_54779


namespace NUMINAMATH_CALUDE_julios_fishing_time_l547_54764

/-- Julio's fishing problem -/
theorem julios_fishing_time (catch_rate : ℕ) (fish_lost : ℕ) (final_fish : ℕ) (h : ℕ) : 
  catch_rate = 7 → fish_lost = 15 → final_fish = 48 → 
  catch_rate * h - fish_lost = final_fish → h = 9 := by
sorry

end NUMINAMATH_CALUDE_julios_fishing_time_l547_54764


namespace NUMINAMATH_CALUDE_exam_marks_theorem_l547_54788

theorem exam_marks_theorem (T : ℝ) 
  (h1 : 0.40 * T + 40 = 160) 
  (h2 : 0.60 * T - 160 = 20) : True :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_theorem_l547_54788


namespace NUMINAMATH_CALUDE_line_intercepts_sum_and_product_l547_54773

/-- Given a line with equation y - 2 = -3(x + 5), prove that the sum of its
    x-intercept and y-intercept is -52/3, and their product is 169/3. -/
theorem line_intercepts_sum_and_product :
  let f : ℝ → ℝ := λ x => -3 * (x + 5) + 2
  let x_intercept := -13 / 3
  let y_intercept := f 0
  (x_intercept + y_intercept = -52 / 3) ∧ (x_intercept * y_intercept = 169 / 3) := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_and_product_l547_54773


namespace NUMINAMATH_CALUDE_simplify_expression_l547_54757

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l547_54757


namespace NUMINAMATH_CALUDE_existence_of_triangle_with_divisible_side_lengths_l547_54783

/-- Given an odd prime p, a positive integer n, and 8 distinct points with integer coordinates
    on a circle of diameter p^n, there exists a triangle formed by three of these points
    such that the square of its side lengths is divisible by p^(n+1). -/
theorem existence_of_triangle_with_divisible_side_lengths
  (p : ℕ) (n : ℕ) (h_p_prime : Nat.Prime p) (h_p_odd : Odd p) (h_n_pos : 0 < n)
  (points : Fin 8 → ℤ × ℤ)
  (h_distinct : Function.Injective points)
  (h_on_circle : ∀ i : Fin 8, (points i).1^2 + (points i).2^2 = (p^n)^2) :
  ∃ i j k : Fin 8, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ m : ℕ, (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points j).1 - (points k).1)^2 + ((points j).2 - (points k).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points k).1 - (points i).1)^2 + ((points k).2 - (points i).2)^2) * m = p^(n+1)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_triangle_with_divisible_side_lengths_l547_54783


namespace NUMINAMATH_CALUDE_system_solution_fractional_solution_l547_54789

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  3 * x + y = 7 ∧ 2 * x - y = 3

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ -1 ∧ x ≠ 1 ∧ 1 / (x + 1) = 1 / (x^2 - 1)

-- Theorem for the system of equations
theorem system_solution :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 1 := by
  sorry

-- Theorem for the fractional equation
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_fractional_solution_l547_54789


namespace NUMINAMATH_CALUDE_even_monotone_function_inequality_l547_54795

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem even_monotone_function_inequality (f : ℝ → ℝ) (m : ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_function_inequality_l547_54795


namespace NUMINAMATH_CALUDE_double_price_profit_percentage_l547_54740

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_rate : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_rate : ℝ) :
  initial_profit_rate = 0.20 →
  initial_selling_price = cost * (1 + initial_profit_rate) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_rate = (new_selling_price - cost) / cost →
  new_profit_rate = 1.40 :=
by sorry

end NUMINAMATH_CALUDE_double_price_profit_percentage_l547_54740


namespace NUMINAMATH_CALUDE_C_on_or_inside_circle_O_l547_54797

-- Define the circle O and points A, B, C
variable (O : ℝ × ℝ) (A B C : ℝ × ℝ)

-- Define the radius of circle O
def radius_O : ℝ := 10

-- Define that A is on circle O
def A_on_circle_O : (A.1 - O.1)^2 + (A.2 - O.2)^2 = radius_O^2 := by sorry

-- Define B as the midpoint of OA
def B_midpoint_OA : B = ((O.1 + A.1)/2, (O.2 + A.2)/2) := by sorry

-- Define the distance between B and C
def BC_distance : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 5^2 := by sorry

-- Theorem to prove
theorem C_on_or_inside_circle_O :
  (C.1 - O.1)^2 + (C.2 - O.2)^2 ≤ radius_O^2 := by sorry

end NUMINAMATH_CALUDE_C_on_or_inside_circle_O_l547_54797


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l547_54776

theorem quadratic_equal_roots :
  ∃ (x : ℝ), x^2 + 2*x + 1 = 0 ∧
  (∀ (y : ℝ), y^2 + 2*y + 1 = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l547_54776


namespace NUMINAMATH_CALUDE_inverse_difference_simplification_l547_54752

theorem inverse_difference_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ - (y / 3)⁻¹) = -(x * y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_inverse_difference_simplification_l547_54752


namespace NUMINAMATH_CALUDE_unique_non_right_triangle_l547_54774

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (7, 24, 26) cannot form a right-angled triangle -/
theorem unique_non_right_triangle :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 8 15 17 ∧
  ¬ is_right_triangle 7 24 26 :=
sorry

end NUMINAMATH_CALUDE_unique_non_right_triangle_l547_54774


namespace NUMINAMATH_CALUDE_min_reciprocal_to_self_reciprocal_81_twice_l547_54735

def reciprocal (x : ℚ) : ℚ := 1 / x

def repeated_reciprocal (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => reciprocal (repeated_reciprocal x n)

theorem min_reciprocal_to_self (x : ℚ) (h : x ≠ 0) :
  ∃ n : ℕ, n > 0 ∧ repeated_reciprocal x n = x ∧
  ∀ m : ℕ, 0 < m ∧ m < n → repeated_reciprocal x m ≠ x :=
by sorry

theorem reciprocal_81_twice :
  ∃ n : ℕ, n = 2 ∧ repeated_reciprocal 81 n = 81 ∧
  ∀ m : ℕ, 0 < m ∧ m < n → repeated_reciprocal 81 m ≠ 81 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_to_self_reciprocal_81_twice_l547_54735


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_specific_point_coordinates_l547_54755

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point2D := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin are the same as its coordinates -/
theorem coordinates_wrt_origin (P : Point2D) : 
  P.x = P.x - origin.x ∧ P.y = P.y - origin.y :=
by sorry

/-- For the specific point P(-2, -4), its coordinates with respect to the origin are (-2, -4) -/
theorem specific_point_coordinates : 
  let P : Point2D := ⟨-2, -4⟩
  P.x - origin.x = -2 ∧ P.y - origin.y = -4 :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_specific_point_coordinates_l547_54755


namespace NUMINAMATH_CALUDE_angle_measure_proof_l547_54751

theorem angle_measure_proof (x : ℝ) (h1 : x + (3 * x + 10) = 90) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l547_54751


namespace NUMINAMATH_CALUDE_abc_inequality_l547_54758

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a*b*c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l547_54758


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l547_54744

theorem arithmetic_calculation : 4 * (8 - 3)^2 / 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l547_54744


namespace NUMINAMATH_CALUDE_zoe_water_bottles_l547_54703

/-- The initial number of water bottles Zoe had in her fridge -/
def initial_bottles : ℕ := 42

/-- The number of bottles Zoe drank -/
def bottles_drank : ℕ := 25

/-- The number of bottles Zoe bought -/
def bottles_bought : ℕ := 30

/-- The final number of bottles Zoe has -/
def final_bottles : ℕ := 47

theorem zoe_water_bottles :
  initial_bottles - bottles_drank + bottles_bought = final_bottles :=
sorry

end NUMINAMATH_CALUDE_zoe_water_bottles_l547_54703


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l547_54708

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l547_54708


namespace NUMINAMATH_CALUDE_expansion_properties_l547_54787

/-- The expansion of (x^(1/4) + x^(3/2))^n where the third-to-last term's coefficient is 45 -/
def expansion (x : ℝ) (n : ℕ) := (x^(1/4) + x^(3/2))^n

/-- The coefficient of the third-to-last term in the expansion -/
def third_to_last_coeff (n : ℕ) := Nat.choose n (n - 2)

theorem expansion_properties (x : ℝ) (n : ℕ) 
  (h : third_to_last_coeff n = 45) : 
  ∃ (k : ℕ), 
    (Nat.choose n k * x^5 = 45 * x^5) ∧ 
    (∀ (j : ℕ), j ≤ n → Nat.choose n j ≤ 252) ∧
    (Nat.choose n 5 * x^(35/4) = 252 * x^(35/4)) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l547_54787


namespace NUMINAMATH_CALUDE_min_value_of_expression_l547_54762

theorem min_value_of_expression (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l547_54762


namespace NUMINAMATH_CALUDE_distribute_5_4_l547_54798

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 identical containers,
    allowing empty containers, is 37. -/
theorem distribute_5_4 : distribute 5 4 = 37 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l547_54798


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l547_54761

/-- Given that N(4,9) is the midpoint of CD and C has coordinates (10,5),
    prove that the sum of the coordinates of D is 11. -/
theorem midpoint_coordinate_sum :
  let N : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (10, 5)
  ∀ D : ℝ × ℝ,
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 + D.2 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l547_54761


namespace NUMINAMATH_CALUDE_average_score_l547_54790

def scores : List ℕ := [65, 67, 76, 82, 85]

theorem average_score : (scores.sum / scores.length : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_score_l547_54790


namespace NUMINAMATH_CALUDE_paper_cup_probability_l547_54727

theorem paper_cup_probability (total_tosses : ℕ) (mouth_up_occurrences : ℕ) 
  (h1 : total_tosses = 200) (h2 : mouth_up_occurrences = 48) :
  (mouth_up_occurrences : ℚ) / total_tosses = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_paper_cup_probability_l547_54727


namespace NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l547_54770

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + 
  (2*n - 3)^2 - (2*n - 5)^2 + (2*n - 5)^2 - (2*n - 7)^2 + 
  (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 24 * n :=
by
  sorry

theorem specific_square_sum_difference : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l547_54770


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l547_54733

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each tray of the first type -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each tray of the second type -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l547_54733


namespace NUMINAMATH_CALUDE_complex_equation_solution_l547_54721

theorem complex_equation_solution (x y : ℝ) (h : (x : ℂ) / (1 + Complex.I) = 1 - y * Complex.I) :
  (x : ℂ) + y * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l547_54721


namespace NUMINAMATH_CALUDE_overlap_length_l547_54747

theorem overlap_length (L D : ℝ) (n : ℕ) (h1 : L = 98) (h2 : D = 83) (h3 : n = 6) :
  ∃ x : ℝ, x = (L - D) / n ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l547_54747


namespace NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l547_54734

-- Define a rectangular cuboid
structure RectangularCuboid where
  vertices : Finset (Fin 8)
  is_cuboid : vertices.card = 8

-- Define a function to count acute-angled triangles
def count_acute_triangles (c : RectangularCuboid) : ℕ :=
  sorry

-- Theorem statement
theorem acute_triangles_in_cuboid (c : RectangularCuboid) :
  count_acute_triangles c = 8 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l547_54734


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l547_54719

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 3

/-- The number of cards in a complete deck of standard playing cards -/
def cards_per_standard_deck : ℕ := 52

/-- The number of incomplete decks of tarot cards -/
def tarot_decks : ℕ := 2

/-- The number of cards in each incomplete tarot deck -/
def cards_per_tarot_deck : ℕ := 72

/-- The number of sets of trading cards -/
def trading_card_sets : ℕ := 5

/-- The number of cards in each trading card set -/
def cards_per_trading_set : ℕ := 100

/-- The number of additional random cards -/
def random_cards : ℕ := 27

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * cards_per_standard_deck + 
  tarot_decks * cards_per_tarot_deck + 
  trading_card_sets * cards_per_trading_set + 
  random_cards

theorem shopkeeper_total_cards : total_cards = 827 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l547_54719


namespace NUMINAMATH_CALUDE_workshop_workers_l547_54767

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 6750) 
  (h2 : technician_salary = 12000) (h3 : other_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
  average_salary * total_workers = 
    num_technicians * technician_salary + 
    (total_workers - num_technicians) * other_salary :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l547_54767


namespace NUMINAMATH_CALUDE_eel_length_problem_l547_54749

theorem eel_length_problem (jenna_eel : ℝ) (bill_eel : ℝ) : 
  jenna_eel = (1/3 : ℝ) * bill_eel → 
  jenna_eel + bill_eel = 64 → 
  jenna_eel = 16 := by
  sorry

end NUMINAMATH_CALUDE_eel_length_problem_l547_54749


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l547_54775

theorem sum_of_squares_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l547_54775


namespace NUMINAMATH_CALUDE_sum_of_numbers_l547_54791

def total_numbers (joyce xavier coraline jayden mickey yvonne : ℕ) : Prop :=
  xavier = 4 * joyce ∧
  coraline = xavier + 50 ∧
  jayden = coraline - 40 ∧
  mickey = jayden + 20 ∧
  yvonne = xavier + joyce ∧
  joyce = 30 ∧
  joyce + xavier + coraline + jayden + mickey + yvonne = 750

theorem sum_of_numbers :
  ∃ (joyce xavier coraline jayden mickey yvonne : ℕ),
    total_numbers joyce xavier coraline jayden mickey yvonne :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l547_54791


namespace NUMINAMATH_CALUDE_balloon_count_total_l547_54742

/-- Calculate the total number of balloons for each color given Sara's and Sandy's balloons -/
theorem balloon_count_total 
  (R1 G1 B1 Y1 O1 R2 G2 B2 Y2 O2 : ℕ) 
  (h1 : R1 = 31) (h2 : G1 = 15) (h3 : B1 = 12) (h4 : Y1 = 18) (h5 : O1 = 10)
  (h6 : R2 = 24) (h7 : G2 = 7)  (h8 : B2 = 14) (h9 : Y2 = 20) (h10 : O2 = 16) :
  R1 + R2 = 55 ∧ 
  G1 + G2 = 22 ∧ 
  B1 + B2 = 26 ∧ 
  Y1 + Y2 = 38 ∧ 
  O1 + O2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_total_l547_54742


namespace NUMINAMATH_CALUDE_decimal_to_base5_conversion_l547_54709

-- Define a function to convert a base-5 number to base-10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the base-5 representation of the number we want to prove
def base5Representation : List Nat := [0, 0, 2, 1]

-- State the theorem
theorem decimal_to_base5_conversion :
  base5ToBase10 base5Representation = 175 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base5_conversion_l547_54709


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruent_to_one_mod_37_l547_54799

theorem smallest_three_digit_congruent_to_one_mod_37 : 
  ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    n % 37 = 1 ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ m % 37 = 1 → n ≤ m) ∧
    n = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruent_to_one_mod_37_l547_54799


namespace NUMINAMATH_CALUDE_equation_solution_l547_54715

theorem equation_solution (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 0 ↔ (x = 2*y ∨ x = -2*y) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l547_54715


namespace NUMINAMATH_CALUDE_power_greater_than_square_l547_54726

theorem power_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_l547_54726


namespace NUMINAMATH_CALUDE_sam_has_46_balloons_l547_54766

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that under the given conditions, Sam has 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_46_balloons_l547_54766


namespace NUMINAMATH_CALUDE_paint_combinations_l547_54743

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- The total number of combinations of color and painting method -/
def total_combinations : ℕ := num_colors * num_methods

theorem paint_combinations : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l547_54743


namespace NUMINAMATH_CALUDE_gumball_probability_l547_54782

/-- Represents a jar of gumballs -/
structure GumballJar where
  blue : ℕ
  pink : ℕ

/-- The probability of drawing a blue gumball -/
def prob_blue (jar : GumballJar) : ℚ :=
  jar.blue / (jar.blue + jar.pink)

/-- The probability of drawing a pink gumball -/
def prob_pink (jar : GumballJar) : ℚ :=
  jar.pink / (jar.blue + jar.pink)

theorem gumball_probability (jar : GumballJar) :
  (prob_blue jar) ^ 2 = 36 / 49 →
  prob_pink jar = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l547_54782


namespace NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l547_54737

/-- Represents a metro line with its one-way travel time -/
structure MetroLine where
  one_way_time : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  red_line : MetroLine
  blue_line : MetroLine
  green_line : MetroLine

/-- Checks if a train returns to its initial position after given minutes -/
def returns_to_initial_position (line : MetroLine) (minutes : ℕ) : Prop :=
  minutes % (2 * line.one_way_time) = 0

/-- The theorem stating that all trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (metro : MetroSystem) :
  returns_to_initial_position metro.red_line 2016 ∧
  returns_to_initial_position metro.blue_line 2016 ∧
  returns_to_initial_position metro.green_line 2016 :=
by
  sorry

/-- The metro system of city N -/
def city_n_metro : MetroSystem :=
  { red_line := { one_way_time := 7 }
  , blue_line := { one_way_time := 8 }
  , green_line := { one_way_time := 9 }
  }

/-- The main theorem proving that all trains in city N's metro return to their initial positions after 2016 minutes -/
theorem city_n_trains_return_after_2016_minutes :
  returns_to_initial_position city_n_metro.red_line 2016 ∧
  returns_to_initial_position city_n_metro.blue_line 2016 ∧
  returns_to_initial_position city_n_metro.green_line 2016 :=
by
  apply all_trains_return_to_initial_positions

end NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l547_54737


namespace NUMINAMATH_CALUDE_simplify_square_root_l547_54769

theorem simplify_square_root (x : ℝ) : 
  Real.sqrt (9 * x^6 + 3 * x^4) = Real.sqrt 3 * x^2 * Real.sqrt (3 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_l547_54769


namespace NUMINAMATH_CALUDE_algebraic_expression_inconsistency_l547_54728

theorem algebraic_expression_inconsistency (a b : ℤ) :
  (-a + b = -1) ∧ (a + b = 5) ∧ (4*a + b = 14) →
  (2*a + b ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_inconsistency_l547_54728


namespace NUMINAMATH_CALUDE_product_of_repeated_digits_l547_54710

def number_of_3s : ℕ := 25
def number_of_6s : ℕ := 25

def number_of_2s : ℕ := 24
def number_of_7s : ℕ := 24

def first_number : ℕ := (3 * (10^number_of_3s - 1)) / 9
def second_number : ℕ := (6 * (10^number_of_6s - 1)) / 9

def result : ℕ := (2 * 10^49 + 10^48 + 7 * (10^24 - 1) / 9) * 10 + 8

theorem product_of_repeated_digits :
  first_number * second_number = result := by sorry

end NUMINAMATH_CALUDE_product_of_repeated_digits_l547_54710


namespace NUMINAMATH_CALUDE_frosting_per_cake_l547_54748

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (total_cans : ℕ) : ℚ :=
  total_cans / (cakes_per_day * days - cakes_eaten)

/-- Theorem stating that given Sara's baking schedule and frosting needs, 
    it takes 2 cans of frosting to frost a single cake -/
theorem frosting_per_cake : 
  cans_per_cake 10 5 12 76 = 2 := by
  sorry

end NUMINAMATH_CALUDE_frosting_per_cake_l547_54748


namespace NUMINAMATH_CALUDE_margarets_mean_score_l547_54760

def scores : List ℝ := [78, 81, 85, 87, 90, 92]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 2 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 84) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 88.5 := by
sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l547_54760


namespace NUMINAMATH_CALUDE_class_test_average_l547_54765

theorem class_test_average (class_size : ℝ) (h_positive : class_size > 0) : 
  let group_a := 0.15 * class_size
  let group_b := 0.50 * class_size
  let group_c := class_size - group_a - group_b
  let score_a := 100
  let score_b := 78
  ∃ score_c : ℝ,
    (group_a * score_a + group_b * score_b + group_c * score_c) / class_size = 76.05 ∧
    score_c = 63 :=
by sorry

end NUMINAMATH_CALUDE_class_test_average_l547_54765


namespace NUMINAMATH_CALUDE_jar_contents_l547_54793

-- Define the number of candy pieces
def candy_pieces : Float := 3409.0

-- Define the number of secret eggs
def secret_eggs : Float := 145.0

-- Define the total number of items
def total_items : Float := candy_pieces + secret_eggs

-- Theorem statement
theorem jar_contents : total_items = 3554.0 := by
  sorry

end NUMINAMATH_CALUDE_jar_contents_l547_54793


namespace NUMINAMATH_CALUDE_last_four_digits_equal_efg_l547_54713

/-- A function that returns the last four digits of a number in base 10 -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- A function that returns the first three digits of a four-digit number -/
def firstThreeDigits (n : ℕ) : ℕ := n / 10

theorem last_four_digits_equal_efg (M : ℕ) (h1 : M > 0) 
  (h2 : lastFourDigits M = lastFourDigits (M^2))
  (h3 : lastFourDigits M ≥ 1000)
  (h4 : (M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0)) :
  firstThreeDigits (lastFourDigits M) = 362 := by
  sorry

#eval firstThreeDigits (lastFourDigits 3626)

end NUMINAMATH_CALUDE_last_four_digits_equal_efg_l547_54713


namespace NUMINAMATH_CALUDE_rectangle_width_equality_l547_54732

theorem rectangle_width_equality (carol_length carol_width jordan_length : ℝ) 
  (h1 : carol_length = 12)
  (h2 : carol_width = 15)
  (h3 : jordan_length = 6)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equality_l547_54732


namespace NUMINAMATH_CALUDE_sin_300_degrees_l547_54714

theorem sin_300_degrees : 
  Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l547_54714


namespace NUMINAMATH_CALUDE_white_circle_area_on_cube_l547_54739

/-- Represents the problem of calculating the area of a white circle on a cube face --/
theorem white_circle_area_on_cube (edge_length : ℝ) (green_paint_area : ℝ) : 
  edge_length = 12 → 
  green_paint_area = 432 → 
  (6 * edge_length^2 - green_paint_area) / 6 = 72 := by
  sorry

#check white_circle_area_on_cube

end NUMINAMATH_CALUDE_white_circle_area_on_cube_l547_54739


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l547_54730

theorem complex_number_quadrant : 
  let z : ℂ := (2 - 3*I) / (I^3)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l547_54730
