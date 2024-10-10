import Mathlib

namespace sequence_is_arithmetic_l3994_399434

/-- Given a sequence a_n with sum of first n terms S_n = n^2 + 1, prove it's arithmetic -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_def : ∀ n : ℕ, S n = n^2 + 1) 
  (sum_relation : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end sequence_is_arithmetic_l3994_399434


namespace calculate_expression_l3994_399463

def A (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

def C (n k : ℕ) : ℕ := n * (n - 1) * (n - 2) / (3 * 2 * 1)

theorem calculate_expression : (3 * A 5 3 + 4 * C 6 3) / (3 * 2 * 1) = 130 / 3 := by
  sorry

end calculate_expression_l3994_399463


namespace two_card_selection_l3994_399494

theorem two_card_selection (deck_size : ℕ) (h : deck_size = 60) : 
  deck_size * (deck_size - 1) = 3540 :=
by sorry

end two_card_selection_l3994_399494


namespace smallest_sum_of_sequence_l3994_399453

theorem smallest_sum_of_sequence (E F G H : ℤ) : 
  E > 0 ∧ F > 0 ∧ G > 0 →  -- E, F, G are positive integers
  ∃ d : ℤ, G - F = F - E ∧ F - E = d →  -- E, F, G form an arithmetic sequence
  ∃ r : ℚ, G = F * r ∧ H = G * r →  -- F, G, H form a geometric sequence
  G / F = 7 / 4 →  -- Given ratio
  ∀ E' F' G' H' : ℤ,
    (E' > 0 ∧ F' > 0 ∧ G' > 0 ∧
     ∃ d' : ℤ, G' - F' = F' - E' ∧ F' - E' = d' ∧
     ∃ r' : ℚ, G' = F' * r' ∧ H' = G' * r' ∧
     G' / F' = 7 / 4) →
    E + F + G + H ≤ E' + F' + G' + H' →
  E + F + G + H = 97 := by
sorry

end smallest_sum_of_sequence_l3994_399453


namespace total_cost_is_2200_l3994_399462

/-- The total cost of buying one smartphone, one personal computer, and one advanced tablet -/
def total_cost (smartphone_price : ℕ) (pc_price_difference : ℕ) : ℕ :=
  let pc_price := smartphone_price + pc_price_difference
  let tablet_price := smartphone_price + pc_price
  smartphone_price + pc_price + tablet_price

/-- Proof that the total cost is $2200 given the specified prices -/
theorem total_cost_is_2200 :
  total_cost 300 500 = 2200 := by
  sorry

end total_cost_is_2200_l3994_399462


namespace greatest_five_digit_with_product_120_sum_18_l3994_399402

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Returns true if the number is five digits -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 120 -/
def N : FiveDigitNumber := sorry

theorem greatest_five_digit_with_product_120_sum_18 :
  isFiveDigit N.val ∧ 
  digitProduct N.val = 120 ∧ 
  (∀ m : FiveDigitNumber, digitProduct m.val = 120 → m.val ≤ N.val) →
  digitSum N.val = 18 := by sorry

end greatest_five_digit_with_product_120_sum_18_l3994_399402


namespace min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l3994_399438

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

/-- The squared magnitude of a vector -/
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

/-- Theorem: Minimum value of |a+tb| and its corresponding t -/
theorem min_magnitude_a_plus_tb :
  (∃ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) = (7 * Real.sqrt 5 / 5)^2) ∧
  (∀ t : ℝ, magnitude_squared (a.1 + t * b.1, a.2 + t * b.2) ≥ (7 * Real.sqrt 5 / 5)^2) ∧
  (magnitude_squared (a.1 + 4/5 * b.1, a.2 + 4/5 * b.2) = (7 * Real.sqrt 5 / 5)^2) :=
sorry

/-- Theorem: Value of t when a-tb is collinear with c -/
theorem collinear_a_minus_tb_c :
  ∃ t : ℝ, t = 3/5 ∧ (a.1 - t * b.1) * c.2 = (a.2 - t * b.2) * c.1 :=
sorry

end min_magnitude_a_plus_tb_collinear_a_minus_tb_c_l3994_399438


namespace square_equal_implies_abs_equal_l3994_399473

theorem square_equal_implies_abs_equal (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

end square_equal_implies_abs_equal_l3994_399473


namespace car_distance_in_30_minutes_l3994_399433

theorem car_distance_in_30_minutes 
  (train_speed : ℝ) 
  (car_speed_ratio : ℝ) 
  (time : ℝ) 
  (h1 : train_speed = 90) 
  (h2 : car_speed_ratio = 2/3) 
  (h3 : time = 1/2) : 
  car_speed_ratio * train_speed * time = 30 := by
  sorry

end car_distance_in_30_minutes_l3994_399433


namespace range_of_f_l3994_399489

open Real

noncomputable def f (x : ℝ) : ℝ := (1 + cos x)^2023 + (1 - cos x)^2023

theorem range_of_f : 
  ∀ y ∈ Set.range (f ∘ (fun x => x * π / 3) ∘ fun t => t * 2 - 1), 2 ≤ y ∧ y ≤ 2^2023 :=
sorry

end range_of_f_l3994_399489


namespace frequency_histogram_interval_length_l3994_399445

/-- Given a frequency histogram interval [a,b), prove that its length |a-b| equals m/h,
    where m is the frequency and h is the histogram height for this interval. -/
theorem frequency_histogram_interval_length
  (a b m h : ℝ)
  (h_interval : a < b)
  (h_frequency : m > 0)
  (h_height : h > 0)
  (h_histogram : h = m / (b - a)) :
  b - a = m / h :=
sorry

end frequency_histogram_interval_length_l3994_399445


namespace brittany_test_average_l3994_399407

def test_average (score1 : ℚ) (score2 : ℚ) : ℚ :=
  (score1 + score2) / 2

theorem brittany_test_average :
  test_average 78 84 = 81 := by
  sorry

end brittany_test_average_l3994_399407


namespace unique_lottery_number_l3994_399496

/-- A five-digit number -/
def FiveDigitNumber := ℕ

/-- Check if a number is a five-digit number -/
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Neighbor's age -/
def neighborAge : ℕ := 45

/-- Theorem: The only five-digit number where the sum of its digits equals 45
    and can be easily solved is 99999 -/
theorem unique_lottery_number :
  ∃! (n : FiveDigitNumber), 
    isFiveDigitNumber n ∧ 
    sumOfDigits n = neighborAge ∧
    (∀ (m : FiveDigitNumber), isFiveDigitNumber m → sumOfDigits m = neighborAge → m = n) :=
by
  sorry

end unique_lottery_number_l3994_399496


namespace inequality_solution_set_l3994_399477

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x < -1 ∨ x > 5 := by
  sorry

end inequality_solution_set_l3994_399477


namespace smallest_resolvable_debt_l3994_399497

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 250) :
  (∃ (d : ℕ), d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) →
  (∃ (d : ℕ), d = 50 ∧ d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) :=
by sorry

end smallest_resolvable_debt_l3994_399497


namespace find_number_l3994_399471

theorem find_number (x n : ℚ) : 
  x = 4 → 
  5 * x + 3 = n * (x - 17) → 
  n = -23 / 13 := by
sorry

end find_number_l3994_399471


namespace break_even_point_correct_l3994_399468

/-- The cost to mold each handle in dollars -/
def moldCost : ℝ := 0.60

/-- The fixed cost to run the molding machine per week in dollars -/
def fixedCost : ℝ := 7640

/-- The selling price per handle in dollars -/
def sellingPrice : ℝ := 4.60

/-- The number of handles needed to break even -/
def breakEvenPoint : ℕ := 1910

/-- Theorem stating that the calculated break-even point is correct -/
theorem break_even_point_correct :
  ↑breakEvenPoint * (sellingPrice - moldCost) = fixedCost :=
sorry

end break_even_point_correct_l3994_399468


namespace quadratic_equal_roots_l3994_399413

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end quadratic_equal_roots_l3994_399413


namespace linear_function_properties_l3994_399448

def f (x : ℝ) := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ (x y : ℝ), f x = y ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0) ∧  -- Fourth quadrant
  (f 2 ≠ 0) ∧                               -- x-intercept is not at (2, 0)
  (∀ x > 0, f x < 2) ∧                      -- When x > 0, y < 2
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)          -- Function is decreasing
  := by sorry

end linear_function_properties_l3994_399448


namespace rancher_loss_rancher_specific_loss_l3994_399404

/-- Calculates the total monetary loss for a rancher given specific conditions --/
theorem rancher_loss (initial_cattle : ℕ) (initial_rate : ℕ) (dead_cattle : ℕ) 
  (sick_cost : ℕ) (reduced_price : ℕ) : ℕ :=
  let expected_revenue := initial_cattle * initial_rate
  let remaining_cattle := initial_cattle - dead_cattle
  let revenue_remaining := remaining_cattle * reduced_price
  let additional_cost := dead_cattle * sick_cost
  let total_loss := (expected_revenue - revenue_remaining) + additional_cost
  total_loss

/-- Proves that the rancher's total monetary loss is $310,500 given the specific conditions --/
theorem rancher_specific_loss : 
  rancher_loss 500 700 350 80 450 = 310500 := by
  sorry

end rancher_loss_rancher_specific_loss_l3994_399404


namespace smallest_special_number_l3994_399409

/-- A number is composite if it's not prime -/
def IsComposite (n : ℕ) : Prop := ¬ Nat.Prime n

/-- A number has no prime factor less than m if all its prime factors are greater than or equal to m -/
def NoPrimeFactorLessThan (n m : ℕ) : Prop :=
  ∀ p, p < m → Nat.Prime p → ¬(p ∣ n)

theorem smallest_special_number : ∃ n : ℕ,
  n > 3000 ∧
  IsComposite n ∧
  ¬(∃ k : ℕ, n = k^2) ∧
  NoPrimeFactorLessThan n 60 ∧
  (∀ m : ℕ, m > 3000 → IsComposite m → ¬(∃ k : ℕ, m = k^2) → NoPrimeFactorLessThan m 60 → m ≥ n) ∧
  n = 4087 := by
sorry

end smallest_special_number_l3994_399409


namespace wheel_radii_theorem_l3994_399414

/-- The ratio of revolutions per minute of wheel A to wheel B -/
def revolution_ratio : ℚ := 1200 / 1500

/-- The total length from the outer radius of wheel A to the outer radius of wheel B in cm -/
def total_length : ℝ := 9

/-- The radius of wheel A in cm -/
def radius_A : ℝ := 2.5

/-- The radius of wheel B in cm -/
def radius_B : ℝ := 2

theorem wheel_radii_theorem :
  revolution_ratio = 4 / 5 ∧
  2 * (radius_A + radius_B) = total_length ∧
  radius_A * 4 = radius_B * 5 := by
  sorry

end wheel_radii_theorem_l3994_399414


namespace curve_equation_and_cosine_value_l3994_399493

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * (3 + Real.sin θ^2) = 12
def C₂ (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

-- Define the condition for α
def α_condition (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (α : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), C₂ A.1 A.2 t₁ α ∧ C₂ B.1 B.2 t₂ α ∧
  (A.1^2 / 4 + A.2^2 / 3 = 1) ∧ (B.1^2 / 4 + B.2^2 / 3 = 1)

-- Define the distance condition
def distance_condition (A B P : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 7/2

theorem curve_equation_and_cosine_value
  (α : ℝ) (A B P : ℝ × ℝ)
  (h_α : α_condition α)
  (h_int : intersection_points A B α)
  (h_dist : distance_condition A B P) :
  (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ (∃ (ρ θ : ℝ), C₁ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
  Real.cos α = 2 * Real.sqrt 7 / 7 :=
sorry

end curve_equation_and_cosine_value_l3994_399493


namespace work_ratio_l3994_399432

/-- Given that A can finish a work in 12 days and A and B together can finish 0.25 part of the work in a day,
    prove that the ratio of time taken by B to finish the work alone to the time taken by A is 1:2 -/
theorem work_ratio (time_A : ℝ) (combined_rate : ℝ) :
  time_A = 12 →
  combined_rate = 0.25 →
  combined_rate = 1 / time_A + 1 / (time_A / 2) :=
by sorry

end work_ratio_l3994_399432


namespace santinos_fruits_l3994_399430

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas produced by each papaya tree -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos produced by each mango tree -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree

theorem santinos_fruits : total_fruits = 80 := by
  sorry

end santinos_fruits_l3994_399430


namespace simplify_fraction_l3994_399437

theorem simplify_fraction : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end simplify_fraction_l3994_399437


namespace x_neq_one_necessary_not_sufficient_l3994_399422

theorem x_neq_one_necessary_not_sufficient :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) :=
by sorry

end x_neq_one_necessary_not_sufficient_l3994_399422


namespace east_bus_speed_l3994_399405

/-- The speed of a bus traveling east, given that it and another bus traveling
    west at 60 mph end up 460 miles apart after 4 hours. -/
theorem east_bus_speed : ℝ := by
  -- Define the speed of the west-traveling bus
  let west_speed : ℝ := 60
  -- Define the time of travel
  let time : ℝ := 4
  -- Define the total distance between buses after travel
  let total_distance : ℝ := 460
  -- Define the speed of the east-traveling bus
  let east_speed : ℝ := (total_distance / time) - west_speed
  -- Assert that the east_speed is equal to 55
  have h : east_speed = 55 := by sorry
  -- Return the speed of the east-traveling bus
  exact east_speed

end east_bus_speed_l3994_399405


namespace quadrilateral_area_l3994_399435

/-- Represents a triangle with its area -/
structure Triangle where
  area : ℝ

/-- Represents the diagram with triangles PQR and XYZ -/
structure Diagram where
  pqr : Triangle
  xyz : Triangle
  smallestTriangleArea : ℝ
  smallestTrianglesCount : ℕ

theorem quadrilateral_area (d : Diagram) 
  (h1 : d.pqr.area = 50)
  (h2 : d.xyz.area = 200)
  (h3 : d.smallestTriangleArea = 1)
  (h4 : d.smallestTrianglesCount = 10)
  : d.xyz.area - d.pqr.area = 150 := by
  sorry

#check quadrilateral_area

end quadrilateral_area_l3994_399435


namespace original_line_length_l3994_399464

/-- Proves that the original length of a line is 1 meter -/
theorem original_line_length
  (erased_length : ℝ)
  (remaining_length : ℝ)
  (h1 : erased_length = 33)
  (h2 : remaining_length = 67)
  (h3 : (100 : ℝ) = (1 : ℝ) * 100) :
  erased_length + remaining_length = 100 := by
sorry

end original_line_length_l3994_399464


namespace division_reduction_l3994_399485

theorem division_reduction (x : ℝ) : (45 / x = 45 - 30) → x = 3 := by
  sorry

end division_reduction_l3994_399485


namespace dividend_calculation_l3994_399416

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 13 → quotient = 17 → remainder = 1 → 
  dividend = divisor * quotient + remainder →
  dividend = 222 := by sorry

end dividend_calculation_l3994_399416


namespace pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l3994_399465

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 - 1 : ℝ) * Complex.I

-- Part 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_3 :
  ∀ m : ℝ, (z m).re = 0 ↔ m = 3 :=
sorry

-- Part 2: z is in the second quadrant iff 1 < m < 3
theorem second_quadrant_iff_m_between_1_and_3 :
  ∀ m : ℝ, ((z m).re < 0 ∧ (z m).im > 0) ↔ (1 < m ∧ m < 3) :=
sorry

end pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l3994_399465


namespace polar_to_rectangular_conversion_l3994_399492

theorem polar_to_rectangular_conversion :
  let r : ℝ := 8
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 4 * Real.sqrt 2) ∧ (y = 4 * Real.sqrt 2) := by sorry

end polar_to_rectangular_conversion_l3994_399492


namespace sally_total_spent_l3994_399423

-- Define the amounts spent on peaches and cherries
def peaches_cost : ℚ := 12.32
def cherries_cost : ℚ := 11.54

-- Define the total cost
def total_cost : ℚ := peaches_cost + cherries_cost

-- Theorem statement
theorem sally_total_spent : total_cost = 23.86 := by
  sorry

end sally_total_spent_l3994_399423


namespace second_number_divisible_by_seven_l3994_399484

theorem second_number_divisible_by_seven (a b c : ℕ+) 
  (ha : a = 105)
  (hc : c = 2436)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 7) :
  7 ∣ b := by
sorry

end second_number_divisible_by_seven_l3994_399484


namespace negative_roots_existence_l3994_399481

theorem negative_roots_existence (p : ℝ) :
  p > 3/5 →
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + 2*p*x₁^3 + p*x₁^2 + x₁^2 + 2*p*x₁ + 1 = 0 ∧
  x₂^4 + 2*p*x₂^3 + p*x₂^2 + x₂^2 + 2*p*x₂ + 1 = 0 :=
by sorry

end negative_roots_existence_l3994_399481


namespace max_ab_perpendicular_lines_l3994_399440

theorem max_ab_perpendicular_lines (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2 * x + (2 * a - 4) * y + 1 = 0 ↔ 2 * b * x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (2 * x₁ + (2 * a - 4) * y₁ + 1 = 0 ∧ 2 * x₂ + (2 * a - 4) * y₂ + 1 = 0 ∧ x₁ ≠ x₂) →
    (2 * b * x₁ + y₁ - 2 = 0 ∧ 2 * b * x₂ + y₂ - 2 = 0 ∧ x₁ ≠ x₂) →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  ∀ c : ℝ, a * b ≤ c → c = 1/2 := by
sorry

end max_ab_perpendicular_lines_l3994_399440


namespace library_shelves_l3994_399427

/-- The number of type C shelves in a library with given conditions -/
theorem library_shelves (total_books : ℕ) (books_per_a : ℕ) (books_per_b : ℕ) (books_per_c : ℕ)
  (percent_a : ℚ) (percent_b : ℚ) (percent_c : ℚ) :
  total_books = 200000 →
  books_per_a = 12 →
  books_per_b = 15 →
  books_per_c = 20 →
  percent_a = 2/5 →
  percent_b = 7/20 →
  percent_c = 1/4 →
  percent_a + percent_b + percent_c = 1 →
  ∃ (shelves_a shelves_b : ℕ),
    ↑shelves_a * books_per_a ≥ ↑total_books * percent_a ∧
    ↑shelves_b * books_per_b ≥ ↑total_books * percent_b ∧
    2500 * books_per_c = ↑total_books * percent_c :=
by sorry


end library_shelves_l3994_399427


namespace triangle_ratio_theorem_l3994_399441

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def PointOnLine (D A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))

-- Define the distance BD
def DistanceBD (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6

-- Define the ratio AD:DC
def RatioADDC (A D C : ℝ × ℝ) : Prop :=
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AD / DC = 18 / 7

-- Theorem statement
theorem triangle_ratio_theorem (A B C D : ℝ × ℝ) :
  Triangle A B C → PointOnLine D A C → DistanceBD B D → RatioADDC A D C :=
by sorry

end triangle_ratio_theorem_l3994_399441


namespace square_plus_inverse_square_equals_six_l3994_399428

theorem square_plus_inverse_square_equals_six (m : ℝ) (h : m^2 - 2*m - 1 = 0) : 
  m^2 + 1/m^2 = 6 := by
  sorry

end square_plus_inverse_square_equals_six_l3994_399428


namespace solution_set_eq_singleton_l3994_399444

/-- The solution set of the system of equations x + y = 1 and x^2 - y^2 = 9 -/
def solution_set : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = 1 ∧ p.1^2 - p.2^2 = 9}

/-- Theorem stating that the solution set contains only the point (5, -4) -/
theorem solution_set_eq_singleton :
  solution_set = {(5, -4)} := by
  sorry

end solution_set_eq_singleton_l3994_399444


namespace book_distribution_l3994_399470

theorem book_distribution (x : ℕ) : x > 0 → (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 :=
  sorry

end book_distribution_l3994_399470


namespace gcd_lcm_product_90_135_l3994_399498

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end gcd_lcm_product_90_135_l3994_399498


namespace discounted_price_approx_l3994_399459

/-- The original price of the shirt in rupees -/
def original_price : ℝ := 746.67

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.25

/-- The discounted price of the shirt -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Theorem stating that the discounted price is approximately 560 rupees -/
theorem discounted_price_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |discounted_price - 560| < ε :=
sorry

end discounted_price_approx_l3994_399459


namespace special_operation_example_l3994_399426

def special_operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem special_operation_example : special_operation 4 2 = 18 := by
  sorry

end special_operation_example_l3994_399426


namespace school_supplies_cost_l3994_399410

/-- Calculates the total cost of school supplies for a class after applying a discount -/
def total_cost_after_discount (num_students : ℕ) 
                               (num_pens num_notebooks num_binders num_highlighters : ℕ)
                               (cost_pen cost_notebook cost_binder cost_highlighter : ℚ)
                               (discount : ℚ) : ℚ :=
  let cost_per_student := num_pens * cost_pen + 
                          num_notebooks * cost_notebook + 
                          num_binders * cost_binder + 
                          num_highlighters * cost_highlighter
  let total_cost := num_students * cost_per_student
  total_cost - discount

/-- Theorem stating the total cost of school supplies after discount -/
theorem school_supplies_cost :
  total_cost_after_discount 30 5 3 1 2 0.5 1.25 4.25 0.75 100 = 260 :=
by sorry

end school_supplies_cost_l3994_399410


namespace survey_students_l3994_399408

theorem survey_students (total_allowance : ℚ) 
  (h1 : total_allowance = 320)
  (h2 : (2 : ℚ) / 3 * 6 + (1 : ℚ) / 3 * 4 = 16 / 3) : 
  ∃ (num_students : ℕ), num_students * (16 : ℚ) / 3 = total_allowance ∧ num_students = 60 := by
  sorry

end survey_students_l3994_399408


namespace min_value_of_expression_equality_condition_l3994_399466

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) = 3 ↔ a = 2 :=
sorry

end min_value_of_expression_equality_condition_l3994_399466


namespace unique_two_digit_number_l3994_399424

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Returns the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- The product of digits of a two-digit number -/
def product_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n * units_digit n

theorem unique_two_digit_number : 
  ∃! (n : TwoDigitNumber), 
    n.val = 4 * sum_of_digits n ∧ 
    n.val = 3 * product_of_digits n ∧
    n.val = 24 := by sorry

end unique_two_digit_number_l3994_399424


namespace parabola_points_theorem_l3994_399450

/-- Parabola passing through given points -/
def parabola (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

theorem parabola_points_theorem :
  ∃ (a c m n : ℝ),
    (parabola a c 0 = -2) ∧
    (parabola a c 1 = 1) ∧
    (parabola a c 2 = m) ∧
    (parabola a c n = -2) ∧
    (a = 2) ∧
    (c = -2) ∧
    (m = 8) ∧
    (n = -1/2) := by
  sorry

end parabola_points_theorem_l3994_399450


namespace marco_marie_age_ratio_l3994_399417

theorem marco_marie_age_ratio :
  ∀ (x : ℕ) (marco_age marie_age : ℕ),
    marie_age = 12 →
    marco_age = x * marie_age + 1 →
    marco_age + marie_age = 37 →
    (marco_age : ℚ) / marie_age = 25 / 12 := by
  sorry

end marco_marie_age_ratio_l3994_399417


namespace fourth_term_is_27_l3994_399483

-- Define the sequence sum function
def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- Theorem statement
theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end fourth_term_is_27_l3994_399483


namespace hydrangea_year_calculation_l3994_399446

/-- The year Lily started buying hydrangeas -/
def start_year : ℕ := 1989

/-- The cost of each hydrangea plant in dollars -/
def plant_cost : ℕ := 20

/-- The total amount Lily has spent on hydrangeas in dollars -/
def total_spent : ℕ := 640

/-- The year up to which Lily has spent the total amount on hydrangeas -/
def end_year : ℕ := 2021

/-- Theorem stating that the calculated end year is correct -/
theorem hydrangea_year_calculation :
  end_year = start_year + (total_spent / plant_cost) :=
by sorry

end hydrangea_year_calculation_l3994_399446


namespace polynomial_identity_sum_l3994_399458

theorem polynomial_identity_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end polynomial_identity_sum_l3994_399458


namespace largest_x_value_l3994_399412

theorem largest_x_value (x : ℝ) : 
  (((17 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 9 * x - 3) ∧ 
   (∀ y : ℝ, ((17 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y = 9 * y - 3) → y ≤ x)) 
  → x = 2/3 := by sorry

end largest_x_value_l3994_399412


namespace least_number_divisible_by_four_primes_ge_5_l3994_399478

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_number_divisible_by_four_primes_ge_5 :
  ∃ (n : Nat) (p₁ p₂ p₃ p₄ : Nat),
    n > 0 ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
    p₁ ≥ 5 ∧ p₂ ≥ 5 ∧ p₃ ≥ 5 ∧ p₄ ≥ 5 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧
    (∀ m : Nat, m > 0 ∧ m < n →
      ¬(∃ (q₁ q₂ q₃ q₄ : Nat),
        is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧
        q₁ ≥ 5 ∧ q₂ ≥ 5 ∧ q₃ ≥ 5 ∧ q₄ ≥ 5 ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
    n = 5005 :=
by sorry

end least_number_divisible_by_four_primes_ge_5_l3994_399478


namespace house_transaction_loss_l3994_399476

/-- Proves that given a house initially valued at $12000, after selling it at a 15% loss
and buying it back at a 20% gain, the original owner loses $240. -/
theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h1 : initial_value = 12000)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.20) :
  initial_value - (initial_value * (1 - loss_percent) * (1 + gain_percent)) = -240 := by
  sorry

end house_transaction_loss_l3994_399476


namespace students_with_d_grade_l3994_399447

/-- Proves that in a course with approximately 600 students, where 1/5 of grades are A's,
    1/4 are B's, 1/2 are C's, and the remaining are D's, the number of students who
    received a D is 30. -/
theorem students_with_d_grade (total_students : ℕ) (a_fraction b_fraction c_fraction : ℚ)
  (h_total : total_students = 600)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_c : c_fraction = 1 / 2)
  (h_sum : a_fraction + b_fraction + c_fraction < 1) :
  total_students - (a_fraction + b_fraction + c_fraction) * total_students = 30 :=
sorry

end students_with_d_grade_l3994_399447


namespace fraction_value_at_four_l3994_399431

theorem fraction_value_at_four : 
  let x : ℝ := 4
  (x^6 - 64*x^3 + 512) / (x^3 - 16) = 48 := by
  sorry

end fraction_value_at_four_l3994_399431


namespace not_right_triangle_l3994_399429

theorem not_right_triangle (a b c : ℚ) (ha : a = 2/3) (hb : b = 2) (hc : c = 5/4) :
  ¬(a^2 + b^2 = c^2) := by sorry

end not_right_triangle_l3994_399429


namespace min_value_a_l3994_399425

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * x * Real.exp x - x - Real.log x ≥ 0) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end min_value_a_l3994_399425


namespace car_travel_time_l3994_399475

/-- Proves that given the conditions of two cars A and B, the time taken by Car B to reach its destination is 1 hour. -/
theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 100 →
  time_A = 6 →
  ratio = 3 →
  (speed_A * time_A) / (speed_B * (speed_A * time_A / (ratio * speed_B))) = 1 := by
  sorry


end car_travel_time_l3994_399475


namespace green_ball_probability_l3994_399467

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of green balls
def green_balls : ℕ := 10

-- Define the probability of drawing a green ball given it's not red
def prob_green_given_not_red : ℚ := green_balls / (total_balls - red_balls)

-- Theorem statement
theorem green_ball_probability :
  prob_green_given_not_red = 2/3 :=
sorry

end green_ball_probability_l3994_399467


namespace jenna_smoothies_l3994_399451

/-- Given that Jenna can make 15 smoothies from 3 strawberries, 
    prove that she can make 90 smoothies from 18 strawberries. -/
theorem jenna_smoothies (smoothies_per_three : ℕ) (strawberries : ℕ) 
  (h1 : smoothies_per_three = 15) 
  (h2 : strawberries = 18) : 
  (smoothies_per_three * strawberries) / 3 = 90 := by
  sorry

end jenna_smoothies_l3994_399451


namespace equation_solution_l3994_399460

theorem equation_solution : ∃! x : ℝ, 4 * x + 9 * x = 430 - 10 * (x + 4) :=
  by
    use 17
    constructor
    · -- Prove that 17 satisfies the equation
      sorry
    · -- Prove that 17 is the unique solution
      sorry

end equation_solution_l3994_399460


namespace visits_neither_country_l3994_399442

/-- Given a group of people and information about their visits to Iceland and Norway,
    calculate the number of people who have visited neither country. -/
theorem visits_neither_country
  (total : ℕ)
  (visited_iceland : ℕ)
  (visited_norway : ℕ)
  (visited_both : ℕ)
  (h_total : total = 90)
  (h_iceland : visited_iceland = 55)
  (h_norway : visited_norway = 33)
  (h_both : visited_both = 51) :
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

#check visits_neither_country

end visits_neither_country_l3994_399442


namespace complex_simplification_l3994_399455

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number -/
noncomputable def z : ℂ := (9 + 2 * i) / (2 + i)

/-- The theorem stating that the given complex number equals 4 - i -/
theorem complex_simplification : z = 4 - i := by sorry

end complex_simplification_l3994_399455


namespace exists_multiple_2020_with_sum_digits_multiple_2020_l3994_399495

/-- Given a natural number n, returns the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a natural number that is a multiple of 2020
    and has a sum of digits that is also a multiple of 2020 -/
theorem exists_multiple_2020_with_sum_digits_multiple_2020 :
  ∃ n : ℕ, 2020 ∣ n ∧ 2020 ∣ sumOfDigits n := by sorry

end exists_multiple_2020_with_sum_digits_multiple_2020_l3994_399495


namespace f_sum_symmetric_max_a_bound_l3994_399469

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp x) / (1 + Real.exp x)

theorem f_sum_symmetric (x : ℝ) : f x + f (-x) = 3 := by sorry

theorem max_a_bound (a : ℝ) :
  (∀ x > 0, f (4 - a * x) + f (x^2) ≥ 3) ↔ a ≤ 4 := by sorry

end f_sum_symmetric_max_a_bound_l3994_399469


namespace moles_of_MgO_formed_l3994_399415

-- Define the chemical elements and compounds
inductive Chemical
| Mg
| CO2
| MgO
| C

-- Define a structure to represent a chemical equation
structure ChemicalEquation :=
  (reactants : List (Chemical × ℕ))
  (products : List (Chemical × ℕ))

-- Define the balanced chemical equation
def balancedEquation : ChemicalEquation :=
  { reactants := [(Chemical.Mg, 2), (Chemical.CO2, 1)]
  , products := [(Chemical.MgO, 2), (Chemical.C, 1)] }

-- Define the available moles of reactants
def availableMg : ℕ := 2
def availableCO2 : ℕ := 1

-- Theorem to prove
theorem moles_of_MgO_formed :
  availableMg = 2 →
  availableCO2 = 1 →
  (balancedEquation.reactants.map (λ (c, n) => (c, n)) = [(Chemical.Mg, 2), (Chemical.CO2, 1)]) →
  (balancedEquation.products.map (λ (c, n) => (c, n)) = [(Chemical.MgO, 2), (Chemical.C, 1)]) →
  ∃ (molesOfMgO : ℕ), molesOfMgO = 2 :=
by
  sorry

end moles_of_MgO_formed_l3994_399415


namespace no_real_solutions_l3994_399474

theorem no_real_solutions :
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end no_real_solutions_l3994_399474


namespace zoes_purchase_cost_l3994_399443

/-- The total cost of Zoe's purchase for herself and her family -/
def total_cost (num_people : ℕ) (soda_cost pizza_cost icecream_cost topping_cost : ℚ) 
  (num_toppings icecream_per_person : ℕ) : ℚ :=
  let soda_total := num_people * soda_cost
  let pizza_total := num_people * (pizza_cost + num_toppings * topping_cost)
  let icecream_total := num_people * icecream_per_person * icecream_cost
  soda_total + pizza_total + icecream_total

/-- Theorem stating that Zoe's total purchase cost is $54.00 -/
theorem zoes_purchase_cost :
  total_cost 6 0.5 1 3 0.75 2 2 = 54 := by
  sorry

end zoes_purchase_cost_l3994_399443


namespace contrapositive_inequality_l3994_399461

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a + c < b + c) → ¬(a < b)) ↔ (a < b → a + c < b + c) :=
by sorry

end contrapositive_inequality_l3994_399461


namespace smallest_b_value_l3994_399421

theorem smallest_b_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b * b = a * c)  -- Geometric progression condition
  (h4 : a + b = 2 * c)  -- Arithmetic progression condition
  : b ≥ 2 ∧ ∃ (a' b' c' : ℤ), a' < b' ∧ b' < c' ∧ b' * b' = a' * c' ∧ a' + b' = 2 * c' ∧ b' = 2 :=
by sorry

#check smallest_b_value

end smallest_b_value_l3994_399421


namespace candy_given_to_janet_and_emily_l3994_399452

-- Define the initial amount of candy
def initial_candy : ℝ := 78.5

-- Define the amount left after giving to Janet
def left_after_janet : ℝ := 68.75

-- Define the amount given to Emily
def given_to_emily : ℝ := 2.25

-- Theorem to prove
theorem candy_given_to_janet_and_emily :
  initial_candy - left_after_janet + given_to_emily = 12 := by
  sorry

end candy_given_to_janet_and_emily_l3994_399452


namespace cookie_shop_purchases_l3994_399480

/-- The number of different types of cookies available. -/
def num_cookies : ℕ := 7

/-- The number of different types of milk available. -/
def num_milk : ℕ := 4

/-- The total number of items Gamma and Delta purchase collectively. -/
def total_items : ℕ := 4

/-- The number of ways Gamma can choose items without repeats. -/
def gamma_choices (k : ℕ) : ℕ := Nat.choose (num_cookies + num_milk) k

/-- The number of ways Delta can choose k cookies with possible repeats. -/
def delta_choices (k : ℕ) : ℕ := 
  (Nat.choose num_cookies k) +  -- All different cookies
  (if k > 1 then num_cookies * (Nat.choose (k - 1 + num_cookies - 1) (num_cookies - 1)) else 0)  -- With repeats

/-- The total number of ways Gamma and Delta can purchase 4 items collectively. -/
def total_ways : ℕ := 
  (gamma_choices 4) +  -- Gamma 4, Delta 0
  (gamma_choices 3) * num_cookies +  -- Gamma 3, Delta 1
  (gamma_choices 2) * (delta_choices 2) +  -- Gamma 2, Delta 2
  (gamma_choices 1) * (delta_choices 3) +  -- Gamma 1, Delta 3
  (delta_choices 4)  -- Gamma 0, Delta 4

theorem cookie_shop_purchases : total_ways = 4096 := by
  sorry

end cookie_shop_purchases_l3994_399480


namespace unique_root_of_sqrt_equation_l3994_399472

theorem unique_root_of_sqrt_equation :
  ∃! x : ℝ, x + 9 ≥ 0 ∧ x - 2 ≥ 0 ∧ Real.sqrt (x + 9) - Real.sqrt (x - 2) = 3 :=
by
  -- The unique solution is x = 19/9
  use 19/9
  sorry

end unique_root_of_sqrt_equation_l3994_399472


namespace reciprocal_sum_pairs_l3994_399420

theorem reciprocal_sum_pairs : 
  ∃! k : ℕ, k > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    Finset.card S = k) :=
by sorry

end reciprocal_sum_pairs_l3994_399420


namespace sum_of_three_hexagons_l3994_399479

theorem sum_of_three_hexagons :
  ∀ (square hexagon : ℚ),
  (3 * square + 2 * hexagon = 18) →
  (2 * square + 3 * hexagon = 20) →
  (3 * hexagon = 72 / 5) :=
by
  sorry

end sum_of_three_hexagons_l3994_399479


namespace minimum_students_l3994_399439

theorem minimum_students (b g : ℕ) : 
  (2 * (b / 2) = 2 * (g * 2 / 3) + 5) →  -- Half of boys equals 2/3 of girls plus 5
  (b ≥ g) →                             -- There are at least as many boys as girls
  (b + g ≥ 17) ∧                        -- The total number of students is at least 17
  (∀ b' g' : ℕ, (2 * (b' / 2) = 2 * (g' * 2 / 3) + 5) → (b' + g' < 17) → (b' < g')) :=
by
  sorry

#check minimum_students

end minimum_students_l3994_399439


namespace second_number_existence_and_uniqueness_l3994_399406

theorem second_number_existence_and_uniqueness :
  ∃! x : ℕ, x > 0 ∧ 220070 = (555 + x) * (2 * (x - 555)) + 70 :=
by sorry

end second_number_existence_and_uniqueness_l3994_399406


namespace min_value_theorem_equality_condition_l3994_399400

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ 2 * Real.sqrt 5 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ 
    a₀^2 + b₀^2 + 1/a₀^2 + 1/b₀^2 + b₀/a₀ = 2 * Real.sqrt 5 := by
  sorry

end min_value_theorem_equality_condition_l3994_399400


namespace rectangle_diagonal_l3994_399491

/-- The diagonal of a rectangle with side lengths 40√3 cm and 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal (a b d : ℝ) (ha : a = 40 * Real.sqrt 3) (hb : b = 30 * Real.sqrt 3) 
  (hd : d ^ 2 = a ^ 2 + b ^ 2) : d = 50 * Real.sqrt 3 := by
  sorry


end rectangle_diagonal_l3994_399491


namespace largest_integer_with_three_digit_square_in_base_7_l3994_399486

theorem largest_integer_with_three_digit_square_in_base_7 :
  ∃ M : ℕ, 
    (∀ n : ℕ, n^2 ≥ 7^2 ∧ n^2 < 7^3 → n ≤ M) ∧
    M^2 ≥ 7^2 ∧ M^2 < 7^3 ∧
    M = 18 :=
by sorry

end largest_integer_with_three_digit_square_in_base_7_l3994_399486


namespace prove_last_score_l3994_399457

def scores : List ℤ := [50, 55, 60, 85, 90, 100]

def is_integer_average (sublist : List ℤ) : Prop :=
  ∃ n : ℤ, (sublist.sum : ℚ) / sublist.length = n

def last_score_is_60 : Prop :=
  ∀ perm : List ℤ, perm.length = 6 →
    perm.toFinset = scores.toFinset →
    (∀ k : ℕ, k ≤ 5 → is_integer_average (perm.take k)) →
    perm.reverse.head? = some 60

theorem prove_last_score : last_score_is_60 := by
  sorry

end prove_last_score_l3994_399457


namespace kiwis_to_add_for_orange_percentage_l3994_399456

/-- Proves that adding 7 kiwis to a box with 24 oranges, 30 kiwis, 15 apples, and 20 bananas
    will make oranges exactly 25% of the total fruits -/
theorem kiwis_to_add_for_orange_percentage (oranges kiwis apples bananas : ℕ) 
    (h1 : oranges = 24) 
    (h2 : kiwis = 30) 
    (h3 : apples = 15) 
    (h4 : bananas = 20) : 
    let total := oranges + kiwis + apples + bananas + 7
    (oranges : ℚ) / total = 1/4 := by sorry

end kiwis_to_add_for_orange_percentage_l3994_399456


namespace women_handshakes_fifteen_couples_l3994_399449

/-- The number of handshakes among women in a group of married couples -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 married couples, if only women shake hands with other women
    (excluding their spouses), the total number of handshakes is 105. -/
theorem women_handshakes_fifteen_couples :
  handshakes 15 = 105 := by
  sorry

end women_handshakes_fifteen_couples_l3994_399449


namespace value_of_x_l3994_399411

theorem value_of_x (w y z x : ℝ) 
  (hw : w = 90)
  (hz : z = 2/3 * w)
  (hy : y = 1/4 * z)
  (hx : x = 1/2 * y) : 
  x = 7.5 := by
  sorry

end value_of_x_l3994_399411


namespace mario_salary_increase_l3994_399401

/-- Proves that Mario's salary increase is 0% given the conditions of the problem -/
theorem mario_salary_increase (mario_salary_this_year : ℝ) 
  (bob_salary_last_year : ℝ) (bob_salary_increase : ℝ) :
  mario_salary_this_year = 4000 →
  bob_salary_last_year = 3 * mario_salary_this_year →
  bob_salary_increase = 0.2 →
  (mario_salary_this_year / bob_salary_last_year * 3 - 1) * 100 = 0 := by
  sorry

end mario_salary_increase_l3994_399401


namespace workshop_2_production_l3994_399488

/-- Represents the production and sampling data for a factory with three workshops -/
structure FactoryData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem workshop_2_production (data : FactoryData) 
    (h_total : data.total_production = 3600)
    (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
    data.sample_2 = 1200 := by
  sorry


end workshop_2_production_l3994_399488


namespace f_min_f_min_range_g_max_min_l3994_399403

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

-- Define the function g(x) = |x-2| + |x-3| - |x-1|
def g (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Theorem stating the minimum value of f(x)
theorem f_min : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 1 :=
sorry

-- Theorem stating the range where f(x) is minimized
theorem f_min_range : ∀ (x : ℝ), f x = 1 → 2 ≤ x ∧ x < 3 :=
sorry

-- Main theorem
theorem g_max_min :
  (∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y) →
  (∃ (a b : ℝ), (∀ (x : ℝ), f x = 1 → g x ≤ a ∧ b ≤ g x) ∧ a = 0 ∧ b = -1) :=
sorry

end f_min_f_min_range_g_max_min_l3994_399403


namespace haleys_marbles_l3994_399487

theorem haleys_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (num_boys : ℕ) 
  (h1 : total_marbles = 28)
  (h2 : marbles_per_boy = 2)
  (h3 : total_marbles = num_boys * marbles_per_boy) :
  num_boys = 14 := by
  sorry

end haleys_marbles_l3994_399487


namespace running_speed_calculation_l3994_399418

/-- Proves that given the specified conditions, the running speed must be 6 mph -/
theorem running_speed_calculation (total_distance : ℝ) (running_time : ℝ) (walking_speed : ℝ) (walking_time : ℝ)
  (h1 : total_distance = 3)
  (h2 : running_time = 20 / 60)
  (h3 : walking_speed = 2)
  (h4 : walking_time = 30 / 60) :
  ∃ (running_speed : ℝ), running_speed * running_time + walking_speed * walking_time = total_distance ∧ running_speed = 6 := by
  sorry

end running_speed_calculation_l3994_399418


namespace increasing_cubic_function_l3994_399490

/-- A function f(x) = x³ + ax - 2 is increasing on [1, +∞) if and only if a ≥ -3 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x - 2)) ↔ a ≥ -3 := by
  sorry

end increasing_cubic_function_l3994_399490


namespace unique_solution_l3994_399419

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : Nat
  h1 : value ≥ 100
  h2 : value ≤ 999

/-- Check if a number has distinct digits in ascending order -/
def hasDistinctAscendingDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 < d2 ∧ d2 < d3

/-- Check if all words in the name of a number start with the same letter -/
def allWordsSameInitial (n : ThreeDigitNumber) : Prop :=
  sorry

/-- Check if a number has identical digits -/
def hasIdenticalDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 = d2 ∧ d2 = d3

/-- Check if all words in the name of a number start with different letters -/
def allWordsDifferentInitials (n : ThreeDigitNumber) : Prop :=
  sorry

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (n1 n2 : ThreeDigitNumber),
    (hasDistinctAscendingDigits n1 ∧ allWordsSameInitial n1) ∧
    (hasIdenticalDigits n2 ∧ allWordsDifferentInitials n2) ∧
    n1.value = 147 ∧ n2.value = 111 := by
  sorry

end unique_solution_l3994_399419


namespace complex_magnitude_power_l3994_399436

theorem complex_magnitude_power : Complex.abs ((3 + 2*Complex.I)^6) = 2197 := by
  sorry

end complex_magnitude_power_l3994_399436


namespace john_is_25_l3994_399499

-- Define John's age and his mother's age
def john_age : ℕ := sorry
def mother_age : ℕ := sorry

-- State the conditions
axiom age_difference : mother_age = john_age + 30
axiom sum_of_ages : john_age + mother_age = 80

-- Theorem to prove
theorem john_is_25 : john_age = 25 := by sorry

end john_is_25_l3994_399499


namespace arithmetic_mean_of_three_numbers_l3994_399454

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 18 ∧ b = 27 ∧ c = 45) : 
  (a + b + c) / 3 = 30 := by
  sorry

end arithmetic_mean_of_three_numbers_l3994_399454


namespace problem_solution_l3994_399482

def f (x : ℝ) := |x| - |2*x - 1|

def M := {x : ℝ | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end problem_solution_l3994_399482
