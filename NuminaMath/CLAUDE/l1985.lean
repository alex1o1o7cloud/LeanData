import Mathlib

namespace cube_surface_area_increase_l1985_198546

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.8 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
sorry

end cube_surface_area_increase_l1985_198546


namespace lcm_problem_l1985_198561

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 18 m = 54) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
sorry

end lcm_problem_l1985_198561


namespace fred_car_wash_earnings_l1985_198520

/-- Fred's earnings from washing the family car -/
def car_wash_earnings (weekly_allowance : ℕ) (final_amount : ℕ) : ℕ :=
  final_amount - weekly_allowance / 2

/-- Proof that Fred earned $6 from washing the family car -/
theorem fred_car_wash_earnings :
  car_wash_earnings 16 14 = 6 :=
by sorry

end fred_car_wash_earnings_l1985_198520


namespace two_year_increase_l1985_198507

/-- Given an initial amount that increases by 1/8th of itself each year,
    calculate the amount after a given number of years. -/
def amount_after_years (initial_amount : ℚ) (years : ℕ) : ℚ :=
  initial_amount * (1 + 1/8) ^ years

/-- Theorem: If an initial amount of 1600 increases by 1/8th of itself each year for two years,
    the final amount will be 2025. -/
theorem two_year_increase : amount_after_years 1600 2 = 2025 := by
  sorry

#eval amount_after_years 1600 2

end two_year_increase_l1985_198507


namespace yang_hui_field_equation_l1985_198580

theorem yang_hui_field_equation (area : ℕ) (length width : ℕ) :
  area = 650 ∧ width = length - 1 →
  length * (length - 1) = area :=
by sorry

end yang_hui_field_equation_l1985_198580


namespace proportion_equality_l1985_198548

theorem proportion_equality (x : ℚ) : 
  (3 : ℚ) / 5 = 12 / 20 ∧ x / 10 = 16 / 40 → x = 4 := by
  sorry

end proportion_equality_l1985_198548


namespace trebled_resultant_l1985_198555

theorem trebled_resultant (initial_number : ℕ) : initial_number = 5 → 
  3 * (2 * initial_number + 15) = 75 := by
  sorry

end trebled_resultant_l1985_198555


namespace inequality_proof_l1985_198512

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c / a + c * a / b + a * b / c ≥ a + b + c) ∧
  (a + b + c = 1 → (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6) := by
  sorry

end inequality_proof_l1985_198512


namespace school_enrollment_problem_l1985_198579

theorem school_enrollment_problem (x y : ℝ) : 
  x + y = 4000 →
  0.07 * x - 0.03 * y = 40 →
  y = 2400 :=
by
  sorry

end school_enrollment_problem_l1985_198579


namespace simplify_trig_expression_l1985_198565

theorem simplify_trig_expression :
  Real.sqrt (1 + 2 * Real.sin (π - 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end simplify_trig_expression_l1985_198565


namespace quadratic_roots_property_l1985_198564

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end quadratic_roots_property_l1985_198564


namespace greatest_multiple_less_than_700_l1985_198501

theorem greatest_multiple_less_than_700 : ∃ n : ℕ, n = 680 ∧ 
  (∀ m : ℕ, m < 700 ∧ 5 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end greatest_multiple_less_than_700_l1985_198501


namespace licorice_probability_l1985_198536

def n : ℕ := 7
def k : ℕ := 5
def p : ℚ := 3/5

theorem licorice_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = 20412/78125 := by
  sorry

end licorice_probability_l1985_198536


namespace equation_solution_l1985_198539

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 2 * (x - 2)^2 - (6 - 3*x)
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end equation_solution_l1985_198539


namespace consecutive_odd_numbers_sum_l1985_198559

theorem consecutive_odd_numbers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 52) →  -- sum condition
  (x = 25) :=  -- conclusion
by sorry

end consecutive_odd_numbers_sum_l1985_198559


namespace equation_solution_l1985_198598

theorem equation_solution :
  ∃! x : ℚ, x ≠ -1 ∧ (x^2 + x + 1) / (x + 1) = x + 2 :=
by
  use (-1/2)
  sorry

end equation_solution_l1985_198598


namespace max_value_of_sum_products_l1985_198531

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d ≤ 10000 := by
sorry

end max_value_of_sum_products_l1985_198531


namespace geometric_sequence_a2_l1985_198570

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/8 := by
sorry

end geometric_sequence_a2_l1985_198570


namespace perpendicular_iff_m_eq_neg_one_l1985_198547

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: x + y = 0 -/
def line1 : Line := ⟨1, 1, 0⟩

/-- The second line: x + my = 0 -/
def line2 (m : ℝ) : Line := ⟨1, m, 0⟩

/-- Theorem: The lines x+y=0 and x+my=0 are perpendicular if and only if m=-1 -/
theorem perpendicular_iff_m_eq_neg_one :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = -1 :=
sorry

end perpendicular_iff_m_eq_neg_one_l1985_198547


namespace mnp_nmp_difference_implies_mmp_nnp_difference_l1985_198505

/-- Represents a three-digit number in base 10 -/
def ThreeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem mnp_nmp_difference_implies_mmp_nnp_difference
  (m n p : ℕ)
  (h : ThreeDigitNumber m n p - ThreeDigitNumber n m p = 180) :
  ThreeDigitNumber m m p - ThreeDigitNumber n n p = 220 := by
  sorry

end mnp_nmp_difference_implies_mmp_nnp_difference_l1985_198505


namespace books_count_proof_l1985_198538

/-- Given a ratio of items and a total count, calculates the number of items for a specific part of the ratio. -/
def calculate_items (ratio : List Nat) (total_items : Nat) (part_index : Nat) : Nat :=
  let total_parts := ratio.sum
  let items_per_part := total_items / total_parts
  items_per_part * (ratio.get! part_index)

/-- Proves that given the ratio 7:3:2 for books, pens, and notebooks, and a total of 600 items, the number of books is 350. -/
theorem books_count_proof :
  let ratio := [7, 3, 2]
  let total_items := 600
  let books_index := 0
  calculate_items ratio total_items books_index = 350 := by
  sorry

end books_count_proof_l1985_198538


namespace square_function_not_property_P_l1985_198556

/-- Property P for a function f --/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2

/-- The square function --/
def square_function (x : ℝ) : ℝ := x^2

/-- Theorem: The square function does not have property P --/
theorem square_function_not_property_P : ¬(has_property_P square_function) := by
  sorry

end square_function_not_property_P_l1985_198556


namespace smallest_n_with_common_factor_l1985_198597

theorem smallest_n_with_common_factor : 
  ∃ (n : ℕ), n > 0 ∧ n = 10 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4)) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (5*n + 4)) :=
sorry

end smallest_n_with_common_factor_l1985_198597


namespace systematic_sample_theorem_l1985_198516

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  interval : Nat
  elements : Finset Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (n : Nat) (s : SystematicSample) : Prop :=
  n ∈ s.elements

theorem systematic_sample_theorem (s : SystematicSample) 
  (h_total : s.total = 52)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = 13)
  (h_6 : in_sample 6 s)
  (h_32 : in_sample 32 s)
  (h_45 : in_sample 45 s) :
  in_sample 19 s := by
  sorry

#check systematic_sample_theorem

end systematic_sample_theorem_l1985_198516


namespace complex_number_quadrant_l1985_198581

theorem complex_number_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i)^2 / (1 + i)
  (z.re < 0 ∧ z.im < 0) := by sorry

end complex_number_quadrant_l1985_198581


namespace coin_flipping_theorem_l1985_198529

theorem coin_flipping_theorem :
  ∀ (initial_state : Fin 2015 → Bool),
  ∃! (final_state : Bool),
    (∀ (i : Fin 2015), final_state = initial_state i) ∨
    (∀ (i : Fin 2015), final_state ≠ initial_state i) :=
by
  sorry


end coin_flipping_theorem_l1985_198529


namespace unique_integer_solution_l1985_198573

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) → m = 4 := by
  sorry

end unique_integer_solution_l1985_198573


namespace polynomial_coefficient_sum_l1985_198594

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  (m = 3 ∨ m = -1) :=
by sorry

end polynomial_coefficient_sum_l1985_198594


namespace two_sided_iced_cubes_count_l1985_198562

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  hasTopIcing : Bool
  hasSideIcing : Bool
  hasVerticalStrip : Bool

/-- Counts the number of 1x1x1 cubes with exactly two iced sides -/
def countTwoSidedIcedCubes (cake : IcedCake) : Nat :=
  sorry

/-- Theorem stating that a 5x5x5 cake with specified icing has 27 two-sided iced cubes -/
theorem two_sided_iced_cubes_count (cake : IcedCake) :
  cake.size = 5 ∧ cake.hasTopIcing ∧ cake.hasSideIcing ∧ cake.hasVerticalStrip →
  countTwoSidedIcedCubes cake = 27 := by
  sorry

end two_sided_iced_cubes_count_l1985_198562


namespace largest_coefficient_term_l1985_198517

theorem largest_coefficient_term (n : ℕ+) :
  ∀ k : ℕ, k ≠ n + 1 →
    Nat.choose (2 * n) (n + 1) ≥ Nat.choose (2 * n) k := by
  sorry

end largest_coefficient_term_l1985_198517


namespace glasses_in_smaller_box_l1985_198503

theorem glasses_in_smaller_box :
  ∀ (x : ℕ) (s l : ℕ),
  -- There are two different-sized boxes
  s = 1 →
  -- There are 16 more larger boxes than smaller boxes
  l = s + 16 →
  -- One box (smaller) contains x glasses, the other (larger) contains 16 glasses
  -- The total number of glasses is 480
  x * s + 16 * l = 480 →
  -- Prove that the number of glasses in the smaller box is 208
  x = 208 := by
sorry

end glasses_in_smaller_box_l1985_198503


namespace company_employees_l1985_198519

/-- The number of employees in a company satisfying certain conditions. -/
theorem company_employees : 
  ∀ (total_females : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_females = 110 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (total_employees : ℕ),
    total_employees = 180 ∧
    total_employees = advanced_degrees + (males_college_only + (total_females - females_advanced)) :=
by sorry

end company_employees_l1985_198519


namespace unique_base_solution_l1985_198510

/-- Convert a number from base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [1, 7, 2] b + toDecimal [1, 5, 6] b = toDecimal [3, 4, 0] b

/-- The main theorem stating that 10 is the unique solution --/
theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equationHolds b :=
  sorry

end unique_base_solution_l1985_198510


namespace store_distance_ratio_l1985_198524

/-- Represents the distances between locations in Jason's commute --/
structure CommuteDistances where
  house_to_first : ℝ
  first_to_second : ℝ
  second_to_third : ℝ
  third_to_work : ℝ

/-- Theorem stating the ratio of distances between stores --/
theorem store_distance_ratio (d : CommuteDistances) :
  d.house_to_first = 4 ∧
  d.first_to_second = 6 ∧
  d.third_to_work = 4 ∧
  d.second_to_third > d.first_to_second ∧
  d.house_to_first + d.first_to_second + d.second_to_third + d.third_to_work = 24 →
  d.second_to_third / d.first_to_second = 5 / 3 :=
by sorry

end store_distance_ratio_l1985_198524


namespace monthly_sales_fraction_l1985_198557

theorem monthly_sales_fraction (december_sales : ℝ) (monthly_sales : ℝ) (total_sales : ℝ) :
  december_sales = 6 * monthly_sales →
  december_sales = 0.35294117647058826 * total_sales →
  monthly_sales = (1 / 17) * total_sales := by
sorry

end monthly_sales_fraction_l1985_198557


namespace manny_marbles_l1985_198592

theorem manny_marbles (total : ℕ) (mario_ratio manny_ratio given : ℕ) : 
  total = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  given = 2 →
  (manny_ratio * (total / (mario_ratio + manny_ratio))) - given = 18 :=
by sorry

end manny_marbles_l1985_198592


namespace second_machine_rate_l1985_198506

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Given two copy machines where the first makes 40 copies per minute,
    and together they make 2850 copies in half an hour,
    prove that the second machine makes 55 copies per minute -/
theorem second_machine_rate 
  (machine1 machine2 : CopyMachine)
  (h1 : machine1.copies_per_minute = 40)
  (h2 : machine1.copies_per_minute * 30 + machine2.copies_per_minute * 30 = 2850) :
  machine2.copies_per_minute = 55 := by
sorry

end second_machine_rate_l1985_198506


namespace cyclist_distance_l1985_198528

theorem cyclist_distance (x t : ℝ) 
  (h1 : (x + 1/3) * (3*t/4) = x * t)
  (h2 : (x - 1/3) * (t + 3) = x * t) :
  x * t = 132 := by
  sorry

end cyclist_distance_l1985_198528


namespace chocolate_boxes_price_l1985_198560

/-- The price of the small box of chocolates -/
def small_box_price : ℝ := 6

/-- The price of the large box of chocolates -/
def large_box_price : ℝ := small_box_price + 3

/-- The total cost of both boxes -/
def total_cost : ℝ := 15

theorem chocolate_boxes_price :
  small_box_price + large_box_price = total_cost ∧
  large_box_price = small_box_price + 3 ∧
  small_box_price = 6 := by
sorry

end chocolate_boxes_price_l1985_198560


namespace inequality_preservation_l1985_198574

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 1 > b - 2 := by
  sorry

end inequality_preservation_l1985_198574


namespace binomial_coefficient_problem_l1985_198533

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by sorry

end binomial_coefficient_problem_l1985_198533


namespace trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l1985_198576

/-- Represents a metro line with a given round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Theorem stating that after 2016 minutes, all trains will be at their starting positions -/
theorem trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  2016 % system.redLine.roundTripTime = 0 ∧
  2016 % system.blueLine.roundTripTime = 0 ∧
  2016 % system.greenLine.roundTripTime = 0 := by
  sorry

/-- Function to check if a train is at its starting position after a given time -/
def isAtStartPosition (line : MetroLine) (time : ℕ) : Bool :=
  time % line.roundTripTime = 0

/-- Theorem stating that all trains are at their starting positions after 2016 minutes -/
theorem all_trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  isAtStartPosition system.redLine 2016 ∧
  isAtStartPosition system.blueLine 2016 ∧
  isAtStartPosition system.greenLine 2016 := by
  sorry

end trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l1985_198576


namespace manuscript_typing_cost_l1985_198569

def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) (twice_revised_set : ℕ) 
                    (thrice_revised : ℕ) (thrice_revised_sets : ℕ) : ℕ :=
  let initial_cost := total_pages * 5
  let once_revised_cost := once_revised * 3
  let twice_revised_cost := (twice_revised - twice_revised_set) * 3 * 2 + twice_revised_set * 3 * 2 + 10
  let thrice_revised_cost := (thrice_revised - thrice_revised_sets * 10) * 3 * 3 + thrice_revised_sets * 15
  initial_cost + once_revised_cost + twice_revised_cost + thrice_revised_cost

theorem manuscript_typing_cost :
  manuscript_cost 200 50 70 10 40 2 = 1730 :=
by
  sorry

end manuscript_typing_cost_l1985_198569


namespace events_mutually_exclusive_and_complementary_l1985_198551

/-- A bag containing balls of two colors -/
structure Bag where
  black : ℕ
  white : ℕ

/-- The event of drawing balls from the bag -/
structure Draw where
  total : ℕ
  black : ℕ
  white : ℕ

/-- Definition of the specific bag in the problem -/
def problem_bag : Bag := { black := 2, white := 2 }

/-- Definition of drawing two balls -/
def two_ball_draw (b : Bag) : Set Draw := 
  {d | d.total = 2 ∧ d.black + d.white = d.total ∧ d.black ≤ b.black ∧ d.white ≤ b.white}

/-- Event: At least one black ball is drawn -/
def at_least_one_black (d : Draw) : Prop := d.black ≥ 1

/-- Event: All drawn balls are white -/
def all_white (d : Draw) : Prop := d.white = d.total

/-- Theorem: The events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  let draw_set := two_ball_draw problem_bag
  ∀ d ∈ draw_set, (at_least_one_black d ↔ ¬ all_white d) ∧ 
                  (at_least_one_black d ∨ all_white d) :=
by sorry

end events_mutually_exclusive_and_complementary_l1985_198551


namespace factorial_difference_quotient_l1985_198550

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end factorial_difference_quotient_l1985_198550


namespace red_then_black_combinations_l1985_198523

def standard_deck : ℕ := 52
def red_cards : ℕ := 26
def black_cards : ℕ := 26

theorem red_then_black_combinations : 
  standard_deck = red_cards + black_cards →
  red_cards * black_cards = 676 := by
  sorry

end red_then_black_combinations_l1985_198523


namespace binomial_probability_problem_l1985_198582

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
def prob_ge (X : BinomialRV) (k : ℕ) : ℝ :=
  sorry

theorem binomial_probability_problem (X Y : BinomialRV) 
  (hX : X.n = 2) (hY : Y.n = 4) (hp : X.p = Y.p)
  (h_prob : prob_ge X 1 = 5/9) : 
  prob_ge Y 2 = 11/27 := by
  sorry

end binomial_probability_problem_l1985_198582


namespace quadratic_function_m_value_l1985_198522

/-- A quadratic function g(x) with integer coefficients -/
def g (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem stating that under given conditions, m = 5 -/
theorem quadratic_function_m_value (a b c m : ℤ) :
  g a b c 2 = 0 →
  70 < g a b c 6 →
  g a b c 6 < 80 →
  110 < g a b c 7 →
  g a b c 7 < 120 →
  2000 * m < g a b c 50 →
  g a b c 50 < 2000 * (m + 1) →
  m = 5 := by sorry

end quadratic_function_m_value_l1985_198522


namespace not_twenty_percent_less_l1985_198541

theorem not_twenty_percent_less (a b : ℝ) (h : a = b * 1.2) : 
  ¬(b = a * 0.8) := by
  sorry

end not_twenty_percent_less_l1985_198541


namespace profit_percentage_change_l1985_198586

def company_profits (revenue2008 : ℝ) : Prop :=
  let profit2008 := 0.1 * revenue2008
  let revenue2009 := 0.8 * revenue2008
  let profit2009 := 0.18 * revenue2009
  let revenue2010 := 1.25 * revenue2009
  let profit2010 := 0.15 * revenue2010
  let profit_change := (profit2010 - profit2008) / profit2008
  profit_change = 0.5

theorem profit_percentage_change (revenue2008 : ℝ) (h : revenue2008 > 0) :
  company_profits revenue2008 := by
  sorry

end profit_percentage_change_l1985_198586


namespace number_calculation_l1985_198521

theorem number_calculation : 
  let x : Float := 0.17999999999999997
  let number : Float := x * 0.05
  number / x = 0.05 ∧ number = 0.009 := by
sorry

end number_calculation_l1985_198521


namespace goods_train_speed_l1985_198549

/-- The speed of a goods train passing a man in an opposite moving train -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 50)
  (h2 : goods_train_length = 280 / 1000)  -- Convert to km
  (h3 : passing_time = 9 / 3600)  -- Convert to hours
  : ∃ (goods_train_speed : ℝ),
    goods_train_speed = 62 ∧
    (man_train_speed + goods_train_speed) * passing_time = goods_train_length :=
by sorry


end goods_train_speed_l1985_198549


namespace point_outside_circle_implies_a_range_l1985_198595

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 = 0

-- Define the condition for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2 ∨ (2 < a ∧ a < 6)

-- Theorem statement
theorem point_outside_circle_implies_a_range :
  ∀ a : ℝ, point_outside_circle 1 1 a → a_range a :=
sorry

end point_outside_circle_implies_a_range_l1985_198595


namespace sufficient_condition_for_f_less_than_one_l1985_198593

theorem sufficient_condition_for_f_less_than_one 
  (a : ℝ) (h_a : a > 1) :
  ∀ x : ℝ, -1 < x ∧ x < 0 → (a * x + 2 * x) < 1 := by
  sorry

end sufficient_condition_for_f_less_than_one_l1985_198593


namespace abs_neg_two_equals_two_l1985_198513

theorem abs_neg_two_equals_two : abs (-2 : ℤ) = 2 := by
  sorry

end abs_neg_two_equals_two_l1985_198513


namespace emily_phone_bill_l1985_198530

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) (data_used : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked - 25) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 15) 0
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- Theorem stating that Emily's phone bill is $59.00 --/
theorem emily_phone_bill : 
  calculate_phone_bill 30 0.1 0.15 5 150 26 16 = 59 := by
  sorry

end emily_phone_bill_l1985_198530


namespace brass_price_is_correct_l1985_198563

/-- The price of copper in dollars per pound -/
def copper_price : ℚ := 65 / 100

/-- The price of zinc in dollars per pound -/
def zinc_price : ℚ := 30 / 100

/-- The total weight of brass in pounds -/
def total_weight : ℚ := 70

/-- The amount of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The amount of zinc used in pounds -/
def zinc_weight : ℚ := total_weight - copper_weight

/-- The selling price of brass per pound -/
def brass_price : ℚ := (copper_price * copper_weight + zinc_price * zinc_weight) / total_weight

theorem brass_price_is_correct : brass_price = 45 / 100 := by
  sorry

end brass_price_is_correct_l1985_198563


namespace row_swap_matrix_l1985_198526

theorem row_swap_matrix (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]
  N * A = !![c, d; a, b] := by sorry

end row_swap_matrix_l1985_198526


namespace expression_simplification_l1985_198577

theorem expression_simplification (a x : ℝ) (h : a ≠ 3*x) :
  1.4 * (3*a^2 + 2*a*x - x^2) / ((3*x + a)*(a + x)) - 2 + 10 * (a*x - 3*x^2) / (a^2 + 9*x^2) = 1 :=
by sorry

end expression_simplification_l1985_198577


namespace max_sum_on_circle_max_sum_achieved_l1985_198558

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 64) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ x + y = 8 := by
  sorry

end max_sum_on_circle_max_sum_achieved_l1985_198558


namespace population_ratio_l1985_198590

/-- The population ratio problem -/
theorem population_ratio 
  (pop_x pop_y pop_z : ℕ) 
  (h1 : pop_x = 7 * pop_y) 
  (h2 : pop_y = 2 * pop_z) : 
  pop_x / pop_z = 14 := by
  sorry

end population_ratio_l1985_198590


namespace max_cross_section_area_l1985_198584

-- Define the prism
def prism_base_side_length : ℝ := 6

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 2 * z = 20

-- Define the cross-section area function
noncomputable def cross_section_area : ℝ := 9

-- Theorem statement
theorem max_cross_section_area :
  ∀ (area : ℝ),
    (∃ (x y z : ℝ), cutting_plane x y z ∧ 
      x^2 + y^2 ≤ (prism_base_side_length / 2)^2) →
    area ≤ cross_section_area :=
by sorry

end max_cross_section_area_l1985_198584


namespace man_sold_portion_l1985_198540

theorem man_sold_portion (lot_value : ℝ) (sold_amount : ℝ) : 
  lot_value = 9200 → 
  sold_amount = 460 → 
  sold_amount / (lot_value / 2) = 1 / 10 := by
sorry

end man_sold_portion_l1985_198540


namespace sqrt_neg_a_rational_implies_a_opposite_perfect_square_l1985_198508

theorem sqrt_neg_a_rational_implies_a_opposite_perfect_square (a : ℝ) :
  (∃ q : ℚ, q^2 = -a) → ∃ n : ℕ, a = -(n^2) := by
  sorry

end sqrt_neg_a_rational_implies_a_opposite_perfect_square_l1985_198508


namespace line_properties_l1985_198596

/-- Represents a line in the form x = my + 1 --/
structure Line where
  m : ℝ

/-- The point (1, 0) is on the line --/
def point_on_line (l : Line) : Prop :=
  1 = l.m * 0 + 1

/-- The area of the triangle formed by the line and the axes when m = 2 --/
def triangle_area (l : Line) : Prop :=
  l.m = 2 → (1 / 2 : ℝ) * 1 * (1 / 2) = (1 / 4 : ℝ)

/-- Main theorem stating that both properties hold for any line of the form x = my + 1 --/
theorem line_properties (l : Line) : point_on_line l ∧ triangle_area l := by
  sorry

end line_properties_l1985_198596


namespace lily_pad_coverage_l1985_198543

/-- Represents the number of days required for the lily pad patch to cover half the lake -/
def days_to_half_coverage : ℕ := 33

/-- Represents the number of days required for the lily pad patch to cover the entire lake -/
def days_to_full_coverage : ℕ := days_to_half_coverage + 1

/-- Theorem stating that the number of days to cover the entire lake is equal to
    the number of days to cover half the lake plus one -/
theorem lily_pad_coverage :
  days_to_full_coverage = days_to_half_coverage + 1 :=
by sorry

end lily_pad_coverage_l1985_198543


namespace sector_central_angle_l1985_198534

/-- Given a circular sector with perimeter 10 and area 4, prove that its central angle is 1/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : 1/2 * l * r = 4) :
  l / r = 1/2 := by
  sorry

end sector_central_angle_l1985_198534


namespace planes_through_skew_line_l1985_198552

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  in_plane : Line → Plane → Prop
  parallel : Plane → Line → Prop
  perpendicular : Plane → Plane → Prop
  skew : Line → Line → Prop

/-- The theorem statement -/
theorem planes_through_skew_line (S : Space3D) 
  (l m : S.Line) (α : S.Plane) 
  (h1 : S.skew l m) 
  (h2 : S.in_plane l α) : 
  (∃ (P : S.Plane), S.parallel P l ∧ ∃ (x : S.Line), S.in_plane x P ∧ x = m) ∧ 
  (∃ (Q : S.Plane), S.perpendicular Q α ∧ ∃ (y : S.Line), S.in_plane y Q ∧ y = m) := by
  sorry

end planes_through_skew_line_l1985_198552


namespace smallest_power_l1985_198583

theorem smallest_power : 2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 := by
  sorry

end smallest_power_l1985_198583


namespace special_sequences_general_terms_l1985_198589

/-- Two sequences of positive real numbers satisfying specific conditions -/
structure SpecialSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  b_pos : ∀ n, b n > 0
  arithmetic : ∀ n, 2 * b n = a n + a (n + 1)
  geometric : ∀ n, (a (n + 1))^2 = b n * b (n + 1)
  initial_a1 : a 1 = 1
  initial_b1 : b 1 = 2
  initial_a2 : a 2 = 3

/-- The general terms of the special sequences -/
theorem special_sequences_general_terms (s : SpecialSequences) :
    (∀ n, s.a n = n * (n + 1) / 2) ∧
    (∀ n, s.b n = (n + 1)^2 / 2) := by
  sorry

end special_sequences_general_terms_l1985_198589


namespace band_arrangement_minimum_band_size_l1985_198571

theorem band_arrangement (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 → n ≥ 168 := by
  sorry

theorem minimum_band_size : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n = 168 := by
  sorry

end band_arrangement_minimum_band_size_l1985_198571


namespace ratio_problem_l1985_198504

theorem ratio_problem (a b : ℝ) : 
  (a / b = 5 / 1) → (a = 45) → (b = 9) := by
  sorry

end ratio_problem_l1985_198504


namespace inequality_of_positive_reals_l1985_198553

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end inequality_of_positive_reals_l1985_198553


namespace grocery_to_gym_speed_l1985_198566

-- Define the constants
def distance_home_to_grocery : ℝ := 840
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the variables
variable (speed_home_to_grocery : ℝ)
variable (speed_grocery_to_gym : ℝ)
variable (time_home_to_grocery : ℝ)
variable (time_grocery_to_gym : ℝ)

-- Define the theorem
theorem grocery_to_gym_speed :
  speed_grocery_to_gym = 2 * speed_home_to_grocery ∧
  time_home_to_grocery = distance_home_to_grocery / speed_home_to_grocery ∧
  time_grocery_to_gym = distance_grocery_to_gym / speed_grocery_to_gym ∧
  time_home_to_grocery = time_grocery_to_gym + time_difference ∧
  speed_home_to_grocery > 0 →
  speed_grocery_to_gym = 30 :=
by sorry

end grocery_to_gym_speed_l1985_198566


namespace selection_theorem_l1985_198585

def number_of_students : ℕ := 10
def number_to_choose : ℕ := 3
def number_of_specific_students : ℕ := 2

def selection_ways : ℕ :=
  Nat.choose (number_of_students - 1) number_to_choose -
  Nat.choose (number_of_students - 1 - number_of_specific_students) number_to_choose

theorem selection_theorem :
  selection_ways = 49 := by
  sorry

end selection_theorem_l1985_198585


namespace symmetric_decreasing_implies_l1985_198599

def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_min_value_on (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

theorem symmetric_decreasing_implies (f : ℝ → ℝ) :
  is_symmetric_about_origin f →
  is_decreasing_on f 1 5 →
  has_min_value_on f 1 5 3 →
  is_decreasing_on f (-5) (-1) ∧ has_min_value_on f (-5) (-1) (-3) :=
sorry

end symmetric_decreasing_implies_l1985_198599


namespace divisibility_condition_l1985_198532

theorem divisibility_condition (m : ℕ) (h1 : m > 2022) 
  (h2 : (2022 + m) ∣ (2022 * m)) : m = 1011 ∨ m = 2022 := by
  sorry

end divisibility_condition_l1985_198532


namespace arithmetic_mean_problem_l1985_198537

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
sorry

end arithmetic_mean_problem_l1985_198537


namespace parabola_circle_tangency_l1985_198542

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : para.eq x y

/-- Circle passing through two points -/
def circle_eq (P₁ P₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - P₁.1) * (x - P₂.1) + (y - P₁.2) * (y - P₂.2) = 0

/-- Main theorem -/
theorem parabola_circle_tangency (para : Parabola) (P₁ P₂ : ParabolaPoint para)
    (h : |P₁.y - P₂.y| = 4 * para.p) :
    ∃! (P : ℝ × ℝ), P ≠ (P₁.x, P₁.y) ∧ P ≠ (P₂.x, P₂.y) ∧
      para.eq P.1 P.2 ∧ circle_eq (P₁.x, P₁.y) (P₂.x, P₂.y) P.1 P.2 :=
  sorry

end parabola_circle_tangency_l1985_198542


namespace jerry_walking_distance_l1985_198545

theorem jerry_walking_distance (monday_miles tuesday_miles : ℝ) 
  (h1 : monday_miles = tuesday_miles)
  (h2 : monday_miles + tuesday_miles = 18) : 
  monday_miles = 9 :=
by sorry

end jerry_walking_distance_l1985_198545


namespace z_in_second_quadrant_l1985_198515

/-- The complex number z = i(1+i) -/
def z : ℂ := Complex.I * (1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Theorem: z is in the second quadrant -/
theorem z_in_second_quadrant : real_part < 0 ∧ imag_part > 0 := by
  sorry

end z_in_second_quadrant_l1985_198515


namespace sin_2alpha_value_l1985_198578

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
sorry

end sin_2alpha_value_l1985_198578


namespace max_log_expression_l1985_198575

theorem max_log_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : 4 * a - 2 * b + 25 * c = 0) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 →
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 := by
sorry

end max_log_expression_l1985_198575


namespace barry_average_l1985_198572

def barry_yards : List ℕ := [98, 107, 85, 89, 91]

theorem barry_average : 
  (barry_yards.sum / barry_yards.length : ℚ) = 94 := by sorry

end barry_average_l1985_198572


namespace triangle_area_l1985_198568

/-- Given a triangle with sides in ratio 5:12:13, perimeter 300 m, and angle 45° between shortest and middle sides, its area is 1500 * √2 m² -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5, 12, 13)) 
  (h_perimeter : a + b + c = 300) (h_angle : Real.cos (45 * π / 180) = b / (2 * a)) : 
  (1/2) * a * b * Real.sin (45 * π / 180) = 1500 * Real.sqrt 2 := by
  sorry

end triangle_area_l1985_198568


namespace unfactorable_quadratic_l1985_198502

/-- A quadratic trinomial that cannot be factored into linear binomials with integer coefficients -/
theorem unfactorable_quadratic (a b c : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eval : a * 1991^2 + b * 1991 + c = p) :
  ¬ ∃ (d₁ d₂ e₁ e₂ : ℤ), ∀ x, a * x^2 + b * x + c = (d₁ * x + e₁) * (d₂ * x + e₂) :=
by sorry

end unfactorable_quadratic_l1985_198502


namespace divisibility_property_l1985_198588

theorem divisibility_property (n : ℕ) : 
  (n - 1) ∣ (n^n - 7*n + 5*n^2024 + 3*n^2 - 2) := by
  sorry

end divisibility_property_l1985_198588


namespace john_number_is_13_l1985_198587

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_is_13 :
  ∃! x : ℕ, is_two_digit x ∧
    92 ≤ switch_digits (4 * x + 17) ∧
    switch_digits (4 * x + 17) ≤ 96 ∧
    x = 13 :=
by sorry

end john_number_is_13_l1985_198587


namespace cubic_root_sum_l1985_198500

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 2) + q/(p*r + 2) + r/(p*q + 2) = 4 + 9/20 := by
sorry

end cubic_root_sum_l1985_198500


namespace divisibility_in_base_greater_than_six_l1985_198535

theorem divisibility_in_base_greater_than_six (a : ℕ) (h : a > 6) :
  ∃ k : ℕ, a^10 + 2*a^9 + 3*a^8 + 4*a^7 + 5*a^6 + 6*a^5 + 5*a^4 + 4*a^3 + 3*a^2 + 2*a + 1
         = k * (a^4 + 2*a^3 + 3*a^2 + 2*a + 1) :=
by sorry

end divisibility_in_base_greater_than_six_l1985_198535


namespace negation_of_existence_negation_of_rational_equation_l1985_198527

theorem negation_of_existence (P : ℚ → Prop) : 
  (¬ ∃ x : ℚ, P x) ↔ (∀ x : ℚ, ¬ P x) := by sorry

theorem negation_of_rational_equation : 
  (¬ ∃ x : ℚ, x - 2 = 0) ↔ (∀ x : ℚ, x - 2 ≠ 0) := by sorry

end negation_of_existence_negation_of_rational_equation_l1985_198527


namespace square_perimeter_l1985_198567

/-- Given two squares I and II, where the diagonal of I is a+b and the area of II is twice the area of I, 
    the perimeter of II is 4(a+b) -/
theorem square_perimeter (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 2 * area_I
  let side_II := Real.sqrt area_II
  side_II * 4 = 4 * (a + b) := by
  sorry

end square_perimeter_l1985_198567


namespace problem_statement_l1985_198518

theorem problem_statement :
  (∃ n : ℤ, 15 = 3 * n) ∧
  (∃ m : ℤ, 121 = 11 * m) ∧ (¬ ∃ k : ℤ, 60 = 11 * k) ∧
  (∃ p : ℤ, 63 = 7 * p) :=
by sorry

end problem_statement_l1985_198518


namespace jan_skips_after_training_l1985_198511

/-- The number of skips Jan does in 5 minutes after doubling her initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Theorem stating that Jan does 700 skips in 5 minutes after doubling her initial speed of 70 skips per minute -/
theorem jan_skips_after_training :
  total_skips 70 5 = 700 := by
  sorry

end jan_skips_after_training_l1985_198511


namespace moon_distance_scientific_notation_l1985_198544

/-- The average distance between the Earth and the Moon in kilometers -/
def moon_distance : ℝ := 384000

/-- Theorem stating that the moon distance in scientific notation is 3.84 × 10^5 -/
theorem moon_distance_scientific_notation : moon_distance = 3.84 * (10 ^ 5) := by
  sorry

end moon_distance_scientific_notation_l1985_198544


namespace polygon_interior_angles_sum_l1985_198554

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n - 2) * 180 = 900 → n = 7 := by
  sorry

end polygon_interior_angles_sum_l1985_198554


namespace bryans_precious_stones_l1985_198509

theorem bryans_precious_stones (price_per_stone : ℕ) (total_amount : ℕ) (h1 : price_per_stone = 1785) (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end bryans_precious_stones_l1985_198509


namespace bus_patrons_count_l1985_198591

/-- The number of patrons a golf cart can fit -/
def golf_cart_capacity : ℕ := 3

/-- The number of patrons who came in cars -/
def car_patrons : ℕ := 12

/-- The number of golf carts needed to transport all patrons -/
def golf_carts_needed : ℕ := 13

/-- The number of patrons who came from a bus -/
def bus_patrons : ℕ := golf_carts_needed * golf_cart_capacity - car_patrons

theorem bus_patrons_count : bus_patrons = 27 := by
  sorry

end bus_patrons_count_l1985_198591


namespace num_kittens_is_eleven_l1985_198525

/-- The number of kittens -/
def num_kittens : ℕ := 11

/-- The weight of the two lightest kittens -/
def weight_lightest : ℕ := 80

/-- The weight of the four heaviest kittens -/
def weight_heaviest : ℕ := 200

/-- The total weight of all kittens -/
def total_weight : ℕ := 500

/-- Theorem stating that the number of kittens is 11 given the weight conditions -/
theorem num_kittens_is_eleven :
  (weight_lightest = 80) →
  (weight_heaviest = 200) →
  (total_weight = 500) →
  (num_kittens = 11) :=
by
  sorry

#check num_kittens_is_eleven

end num_kittens_is_eleven_l1985_198525


namespace power_product_l1985_198514

theorem power_product (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 := by
  sorry

end power_product_l1985_198514
