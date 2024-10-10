import Mathlib

namespace books_read_l3967_396768

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (books_to_read : ℕ) (h1 : total_books = 22) (h2 : books_to_read = 10) :
  total_books - books_to_read = 12 := by
  sorry

#check books_read

end books_read_l3967_396768


namespace problem_solution_l3967_396781

theorem problem_solution (a : ℝ) : a = 1 / (Real.sqrt 2 - 1) → 4 * a^2 - 8 * a + 1 = 5 := by
  sorry

end problem_solution_l3967_396781


namespace algebraic_identities_l3967_396764

theorem algebraic_identities :
  (∃ (x : ℝ), x^2 = 3 ∧ x > 0) ∧ 
  (∃ (y : ℝ), y^2 = 2 ∧ y > 0) →
  (3 * Real.sqrt 3 - (Real.sqrt 12 + Real.sqrt (1/3)) = 2 * Real.sqrt 3 / 3) ∧
  ((1 + Real.sqrt 2) * (2 - Real.sqrt 2) = Real.sqrt 2) := by
  sorry

end algebraic_identities_l3967_396764


namespace complex_modulus_one_l3967_396745

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l3967_396745


namespace existence_of_mn_l3967_396733

theorem existence_of_mn (k : ℕ+) : 
  (∃ m n : ℕ+, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
sorry

end existence_of_mn_l3967_396733


namespace river_depth_l3967_396747

theorem river_depth (depth_may : ℝ) (depth_june : ℝ) (depth_july : ℝ) 
  (h1 : depth_june = depth_may + 10)
  (h2 : depth_july = 3 * depth_june)
  (h3 : depth_july = 45) : 
  depth_may = 5 := by
sorry

end river_depth_l3967_396747


namespace jacket_final_price_l3967_396774

/-- The final price of a jacket after applying two successive discounts --/
theorem jacket_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 250 → 
  discount1 = 0.4 → 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 112.5 := by
  sorry


end jacket_final_price_l3967_396774


namespace g_is_zero_l3967_396732

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (2 * (Real.sin x)^4 + 3 * (Real.cos x)^2) - 
  Real.sqrt (2 * (Real.cos x)^4 + 3 * (Real.sin x)^2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end g_is_zero_l3967_396732


namespace kangaroo_hops_l3967_396725

/-- The distance covered in a single hop, given the remaining distance -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The sum of distances covered in n hops -/
def total_distance (n : ℕ) : ℚ :=
  (1 - (3/4)^n) / (1/4)

/-- The theorem stating that after 6 hops, the total distance covered is 3367/4096 -/
theorem kangaroo_hops : total_distance 6 = 3367 / 4096 := by sorry

end kangaroo_hops_l3967_396725


namespace average_score_is_94_l3967_396706

def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

def total_score : ℕ := june_score + patty_score + josh_score + henry_score
def num_children : ℕ := 4

theorem average_score_is_94 : total_score / num_children = 94 := by
  sorry

end average_score_is_94_l3967_396706


namespace arithmetic_mean_of_fractions_l3967_396735

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 6 / 7
  let c := 4 / 5
  let arithmetic_mean := (a + b) / 2
  (arithmetic_mean ≠ c) ∧ (arithmetic_mean = 51 / 70) := by
  sorry

end arithmetic_mean_of_fractions_l3967_396735


namespace no_solution_implies_b_bounded_l3967_396785

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, ¬(a * Real.cos x + b * Real.cos (3 * x) > 1)) →
  |b| ≤ 1 := by
  sorry

end no_solution_implies_b_bounded_l3967_396785


namespace black_beads_count_l3967_396713

theorem black_beads_count (white_beads : ℕ) (total_pulled : ℕ) :
  white_beads = 51 →
  total_pulled = 32 →
  ∃ (black_beads : ℕ),
    (1 : ℚ) / 6 * black_beads + (1 : ℚ) / 3 * white_beads = total_pulled ∧
    black_beads = 90 :=
by sorry

end black_beads_count_l3967_396713


namespace tangent_length_to_circle_l3967_396759

def origin : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (10, 25)

theorem tangent_length_to_circle (circle : Set (ℝ × ℝ)) 
  (h1 : A ∈ circle) (h2 : B ∈ circle) (h3 : C ∈ circle) :
  ∃ T ∈ circle, ‖T - origin‖ = Real.sqrt 82 ∧ 
  ∀ P ∈ circle, P ≠ T → ‖P - origin‖ > Real.sqrt 82 := by
  sorry

end tangent_length_to_circle_l3967_396759


namespace michelle_savings_denomination_l3967_396704

/-- Given a total savings amount and a number of bills, calculate the denomination of each bill. -/
def billDenomination (totalSavings : ℕ) (numBills : ℕ) : ℕ :=
  totalSavings / numBills

/-- Theorem: Given Michelle's total savings of $800 and 8 bills, the denomination of each bill is $100. -/
theorem michelle_savings_denomination :
  billDenomination 800 8 = 100 := by
  sorry

end michelle_savings_denomination_l3967_396704


namespace percentage_of_sikh_boys_l3967_396720

/-- Given a school with the following demographics:
  - Total number of boys: 850
  - 44% are Muslims
  - 28% are Hindus
  - 153 boys belong to other communities
  Prove that 10% of the boys are Sikhs -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 153 →
  (total - (muslim_percent * total + hindu_percent * total + other : ℚ)) / total = 1 / 10 := by
  sorry

end percentage_of_sikh_boys_l3967_396720


namespace total_laundry_time_is_344_l3967_396717

/-- Represents a load of laundry with washing and drying times -/
structure LaundryLoad where
  washTime : ℕ
  dryTime : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalLoadTime (load : LaundryLoad) : ℕ :=
  load.washTime + load.dryTime

/-- Calculates the total time for all loads of laundry -/
def totalLaundryTime (loads : List LaundryLoad) : ℕ :=
  loads.map totalLoadTime |>.sum

/-- Theorem: The total laundry time for the given loads is 344 minutes -/
theorem total_laundry_time_is_344 :
  let whites : LaundryLoad := { washTime := 72, dryTime := 50 }
  let darks : LaundryLoad := { washTime := 58, dryTime := 65 }
  let colors : LaundryLoad := { washTime := 45, dryTime := 54 }
  let allLoads : List LaundryLoad := [whites, darks, colors]
  totalLaundryTime allLoads = 344 := by
  sorry


end total_laundry_time_is_344_l3967_396717


namespace count_divisors_multiple_of_five_l3967_396708

/-- The number of positive divisors of 7560 that are multiples of 5 -/
def divisors_multiple_of_five : ℕ :=
  (Finset.range 4).card * (Finset.range 4).card * 1 * (Finset.range 2).card

theorem count_divisors_multiple_of_five :
  7560 = 2^3 * 3^3 * 5^1 * 7^1 →
  divisors_multiple_of_five = 32 := by
sorry

end count_divisors_multiple_of_five_l3967_396708


namespace conic_eccentricity_l3967_396773

theorem conic_eccentricity (m : ℝ) : 
  (m = Real.sqrt (2 * 8) ∨ m = -Real.sqrt (2 * 8)) →
  let e := if m > 0 
    then Real.sqrt 3 / 2 
    else Real.sqrt 5
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
    (∀ x y : ℝ, x^2 + y^2/m = 1 ↔ (x/a)^2 + (y/b)^2 = 1) ∧
    e = Real.sqrt (|a^2 - b^2|) / max a b :=
by sorry

end conic_eccentricity_l3967_396773


namespace max_value_of_complex_difference_l3967_396731

theorem max_value_of_complex_difference (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W - (3 - 4*I)) ≤ max_val :=
sorry

end max_value_of_complex_difference_l3967_396731


namespace exists_set_with_square_diff_divides_product_l3967_396786

theorem exists_set_with_square_diff_divides_product (n : ℕ+) :
  ∃ (S : Finset ℕ+), 
    S.card = n ∧ 
    ∀ (a b : ℕ+), a ∈ S → b ∈ S → (a - b)^2 ∣ (a * b) :=
sorry

end exists_set_with_square_diff_divides_product_l3967_396786


namespace t_formula_l3967_396743

theorem t_formula (S₁ S₂ t u : ℝ) (hu : u ≠ 0) (heq : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
sorry

end t_formula_l3967_396743


namespace birds_on_fence_l3967_396740

theorem birds_on_fence (initial_birds additional_birds : ℕ) :
  initial_birds = 12 → additional_birds = 8 →
  initial_birds + additional_birds = 20 := by
  sorry

end birds_on_fence_l3967_396740


namespace problem_1_l3967_396752

theorem problem_1 : -2 - |(-2)| = -4 := by sorry

end problem_1_l3967_396752


namespace other_number_proof_l3967_396757

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 5040)
  (h2 : Nat.gcd a b = 24)
  (h3 : a = 240) :
  b = 504 := by
  sorry

end other_number_proof_l3967_396757


namespace certain_number_is_sixty_l3967_396796

theorem certain_number_is_sixty : 
  ∃ x : ℝ, (10 + 20 + x) / 3 = (10 + 40 + 25) / 3 + 5 → x = 60 := by
  sorry

end certain_number_is_sixty_l3967_396796


namespace grocers_sales_problem_l3967_396756

/-- Proof of the grocer's sales problem -/
theorem grocers_sales_problem
  (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 6435)
  (h2 : sales 1 = 6927)
  (h3 : sales 2 = 6855)
  (h5 : sales 4 = 6562)
  (h6 : sales 5 = 7991)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7000) :
  sales 3 = 7230 := by
  sorry


end grocers_sales_problem_l3967_396756


namespace stack_map_a_front_view_l3967_396738

/-- Represents a column of stacks in the Stack Map --/
def Column := List Nat

/-- Represents the Stack Map A --/
def StackMapA : List Column := [
  [3, 1],       -- First column
  [2, 2, 1],    -- Second column
  [1, 4, 2],    -- Third column
  [5]           -- Fourth column
]

/-- Calculates the front view of a Stack Map --/
def frontView (map : List Column) : List Nat :=
  map.map (List.foldl max 0)

/-- Theorem: The front view of Stack Map A is [3, 2, 4, 5] --/
theorem stack_map_a_front_view :
  frontView StackMapA = [3, 2, 4, 5] := by
  sorry

end stack_map_a_front_view_l3967_396738


namespace simplify_trig_expression_l3967_396707

theorem simplify_trig_expression :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by
  sorry

end simplify_trig_expression_l3967_396707


namespace least_three_digit_7_heavy_l3967_396749

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- A number is three-digit if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  is_three_digit 103 ∧ 
  is_7_heavy 103 ∧ 
  ∀ n : ℕ, is_three_digit n → is_7_heavy n → 103 ≤ n :=
by sorry

end least_three_digit_7_heavy_l3967_396749


namespace arithmetic_sequence_property_l3967_396719

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 2) : 
  a 4 - a 5 + a 6 = 2 := by
sorry

end arithmetic_sequence_property_l3967_396719


namespace sixty_percent_high_profit_puppies_l3967_396795

/-- Represents a litter of puppies with their spot counts -/
structure PuppyLitter where
  total : Nat
  fiveSpots : Nat
  fourSpots : Nat
  twoSpots : Nat

/-- Calculates the percentage of puppies with more than 4 spots -/
def percentageHighProfitPuppies (litter : PuppyLitter) : Rat :=
  (litter.fiveSpots : Rat) / (litter.total : Rat) * 100

/-- The theorem stating that for the given litter, 60% of puppies can be sold for greater profit -/
theorem sixty_percent_high_profit_puppies (litter : PuppyLitter)
    (h1 : litter.total = 10)
    (h2 : litter.fiveSpots = 6)
    (h3 : litter.fourSpots = 3)
    (h4 : litter.twoSpots = 1)
    (h5 : litter.fiveSpots + litter.fourSpots + litter.twoSpots = litter.total) :
    percentageHighProfitPuppies litter = 60 := by
  sorry

end sixty_percent_high_profit_puppies_l3967_396795


namespace fraction_equivalence_l3967_396763

theorem fraction_equivalence (a1 a2 b1 b2 : ℝ) :
  (∀ x : ℝ, x + a2 ≠ 0 → (x + a1) / (x + a2) = b1 / b2) ↔ (b2 = b1 ∧ b1 * a2 = a1 * b2) :=
sorry

end fraction_equivalence_l3967_396763


namespace inequality_proof_l3967_396714

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end inequality_proof_l3967_396714


namespace bill_animals_l3967_396710

-- Define the number of rats
def num_rats : ℕ := 60

-- Define the relationship between rats and chihuahuas
def num_chihuahuas : ℕ := num_rats / 6

-- Define the total number of animals
def total_animals : ℕ := num_rats + num_chihuahuas

-- Theorem to prove
theorem bill_animals : total_animals = 70 := by
  sorry

end bill_animals_l3967_396710


namespace phi_expression_l3967_396783

/-- A function that is directly proportional to x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function that is inversely proportional to x -/
def InverselyProportional (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

/-- The main theorem -/
theorem phi_expression (f g φ : ℝ → ℝ) 
    (h1 : DirectlyProportional f)
    (h2 : InverselyProportional g)
    (h3 : ∀ x : ℝ, φ x = f x + g x)
    (h4 : φ (1/3) = 16)
    (h5 : φ 1 = 8) :
    ∀ x : ℝ, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end phi_expression_l3967_396783


namespace sale_discount_theorem_l3967_396721

/-- Calculates the final amount paid after applying a discount based on the purchase amount -/
def final_amount_paid (initial_amount : ℕ) (discount_per_hundred : ℕ) : ℕ :=
  initial_amount - (initial_amount / 100) * discount_per_hundred

/-- Theorem stating that for a $250 purchase with $10 off per $100 spent, the final amount is $230 -/
theorem sale_discount_theorem :
  final_amount_paid 250 10 = 230 := by
  sorry

end sale_discount_theorem_l3967_396721


namespace car_value_decrease_l3967_396702

/-- Given a car with an initial value and a value after a certain number of years,
    calculate the annual decrease in value. -/
def annual_decrease (initial_value : ℕ) (final_value : ℕ) (years : ℕ) : ℕ :=
  (initial_value - final_value) / years

theorem car_value_decrease :
  let initial_value : ℕ := 20000
  let final_value : ℕ := 14000
  let years : ℕ := 6
  annual_decrease initial_value final_value years = 1000 := by
  sorry

end car_value_decrease_l3967_396702


namespace systematic_sample_theorem_l3967_396791

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the seat numbers in a systematic sample -/
def generate_sample (s : SystematicSample) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.first_element + i * s.interval)

theorem systematic_sample_theorem (sample : SystematicSample)
  (h1 : sample.population_size = 48)
  (h2 : sample.sample_size = 4)
  (h3 : sample.interval = sample.population_size / sample.sample_size)
  (h4 : sample.first_element = 6)
  (h5 : 30 ∈ generate_sample sample)
  (h6 : 42 ∈ generate_sample sample) :
  18 ∈ generate_sample sample :=
sorry

end systematic_sample_theorem_l3967_396791


namespace james_recovery_time_l3967_396748

def initial_healing_time : ℝ := 4

def skin_graft_healing_time (t : ℝ) : ℝ := t * 1.5

def total_recovery_time (t : ℝ) : ℝ := t + skin_graft_healing_time t

theorem james_recovery_time :
  total_recovery_time initial_healing_time = 10 := by
  sorry

end james_recovery_time_l3967_396748


namespace ellipse_k_range_l3967_396746

/-- The range of k for an ellipse with equation x^2/4 + y^2/k = 1 and eccentricity e ∈ (1/2, 1) -/
theorem ellipse_k_range (e : ℝ) (h1 : 1/2 < e) (h2 : e < 1) :
  ∀ k : ℝ, (∃ x y : ℝ, x^2/4 + y^2/k = 1 ∧ e^2 = 1 - (min 4 k)/(max 4 k)) ↔
  (k ∈ Set.Ioo 0 3 ∪ Set.Ioi (16/3)) :=
by sorry

end ellipse_k_range_l3967_396746


namespace perfume_price_decrease_l3967_396778

theorem perfume_price_decrease (original_price increased_price final_price : ℝ) : 
  original_price = 1200 →
  increased_price = original_price * 1.1 →
  final_price = original_price - 78 →
  (increased_price - final_price) / increased_price = 0.15 := by
sorry

end perfume_price_decrease_l3967_396778


namespace quadratic_root_sum_l3967_396775

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 5 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 5 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 3/2) := by
sorry

end quadratic_root_sum_l3967_396775


namespace angle_measure_l3967_396779

theorem angle_measure (α : Real) (h : α > 0 ∧ α < π/2) :
  1 / Real.sqrt (Real.tan (α/2)) = Real.sqrt (2 * Real.sqrt 3) * Real.sqrt (Real.tan (π/18)) + Real.sqrt (Real.tan (α/2)) →
  α = π/3.6 := by
  sorry

end angle_measure_l3967_396779


namespace equilateral_triangle_perimeter_l3967_396789

/-- The perimeter of an equilateral triangle with side length 23 centimeters is 69 centimeters. -/
theorem equilateral_triangle_perimeter :
  ∀ (triangle : Set ℝ × Set ℝ),
    (∀ side : ℝ, side ∈ (triangle.1 ∪ triangle.2) → side = 23) →
    (∃ (a b c : ℝ), a ∈ triangle.1 ∧ b ∈ triangle.1 ∧ c ∈ triangle.2 ∧
      a + b + c = 69) :=
by sorry

end equilateral_triangle_perimeter_l3967_396789


namespace daily_pay_rate_is_twenty_l3967_396755

/-- Calculates the daily pay rate given the total days, worked days, forfeit amount, and net earnings -/
def calculate_daily_pay_rate (total_days : ℕ) (worked_days : ℕ) (forfeit_amount : ℚ) (net_earnings : ℚ) : ℚ :=
  let idle_days := total_days - worked_days
  let total_forfeit := idle_days * forfeit_amount
  (net_earnings + total_forfeit) / worked_days

/-- Theorem stating that given the specified conditions, the daily pay rate is $20 -/
theorem daily_pay_rate_is_twenty :
  calculate_daily_pay_rate 25 23 5 450 = 20 := by
  sorry

end daily_pay_rate_is_twenty_l3967_396755


namespace complex_product_with_i_l3967_396766

theorem complex_product_with_i : (Complex.I * (-1 + 3 * Complex.I)) = (-3 - Complex.I) := by
  sorry

end complex_product_with_i_l3967_396766


namespace show_length_l3967_396758

/-- Proves that the length of each show is 50 minutes given the conditions -/
theorem show_length (gina_choice_ratio : ℝ) (total_shows : ℕ) (gina_minutes : ℝ) 
  (h1 : gina_choice_ratio = 3)
  (h2 : total_shows = 24)
  (h3 : gina_minutes = 900) : 
  (gina_minutes / (gina_choice_ratio * total_shows / (gina_choice_ratio + 1))) = 50 := by
  sorry


end show_length_l3967_396758


namespace isosceles_triangle_m_condition_l3967_396718

/-- Represents an isosceles triangle ABC with side BC of length 8 and sides AB and AC as roots of x^2 - 10x + m = 0 --/
structure IsoscelesTriangle where
  m : ℝ
  ab_ac_roots : ∀ x : ℝ, x^2 - 10*x + m = 0 → (x = ab ∨ x = ac)
  isosceles : ab = ac
  bc_length : bc = 8

/-- The value of m in an isosceles triangle satisfies one of two conditions --/
theorem isosceles_triangle_m_condition (t : IsoscelesTriangle) :
  (∃ x : ℝ, x^2 - 10*x + t.m = 0 ∧ (∀ y : ℝ, y^2 - 10*y + t.m = 0 → y = x)) ∨
  (8^2 - 10*8 + t.m = 0) :=
sorry

end isosceles_triangle_m_condition_l3967_396718


namespace inequality_system_solution_l3967_396767

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) :=
by sorry

end inequality_system_solution_l3967_396767


namespace symmetry_x_axis_example_l3967_396716

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (-2, 1, 5) with respect to the x-axis
    has coordinates (-2, -1, -5) -/
theorem symmetry_x_axis_example : 
  symmetry_x_axis { x := -2, y := 1, z := 5 } = { x := -2, y := -1, z := -5 } := by
  sorry

end symmetry_x_axis_example_l3967_396716


namespace problem_solution_l3967_396797

/-- Given that (k-1)x^|k| + 3 ≥ 0 is a one-variable linear inequality about x and (k-1) ≠ 0, prove that k = -1 -/
theorem problem_solution (k : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^(|k|) + 3 = a * x + b) → -- Linear inequality condition
  (k - 1 ≠ 0) →                                     -- Non-zero coefficient condition
  k = -1 :=
by sorry

end problem_solution_l3967_396797


namespace line_plane_perpendicularity_l3967_396772

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicularLines a b :=
sorry

end line_plane_perpendicularity_l3967_396772


namespace sum_of_five_and_seven_l3967_396701

theorem sum_of_five_and_seven : 5 + 7 = 12 := by
  sorry

end sum_of_five_and_seven_l3967_396701


namespace prove_max_value_l3967_396770

def max_value_theorem (a b c : ℝ × ℝ) : Prop :=
  let norm_squared := λ v : ℝ × ℝ => v.1^2 + v.2^2
  norm_squared a = 9 ∧ 
  norm_squared b = 4 ∧ 
  norm_squared c = 16 →
  norm_squared (a.1 - 3*b.1, a.2 - 3*b.2) + 
  norm_squared (b.1 - 3*c.1, b.2 - 3*c.2) + 
  norm_squared (c.1 - 3*a.1, c.2 - 3*a.2) ≤ 428

theorem prove_max_value : ∀ a b c : ℝ × ℝ, max_value_theorem a b c := by
  sorry

end prove_max_value_l3967_396770


namespace marble_distribution_l3967_396705

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 30 →
  num_friends = 5 →
  total_marbles = num_friends * marbles_per_friend →
  marbles_per_friend = 6 :=
by
  sorry

end marble_distribution_l3967_396705


namespace no_additional_coins_needed_l3967_396751

/-- The minimum number of additional coins needed given the number of friends and initial coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := num_friends * (num_friends + 1) / 2
  if initial_coins ≥ required_coins then 0
  else required_coins - initial_coins

/-- Theorem stating that for 15 friends and 120 initial coins, no additional coins are needed. -/
theorem no_additional_coins_needed :
  min_additional_coins 15 120 = 0 := by
  sorry

end no_additional_coins_needed_l3967_396751


namespace triangle_perimeter_inside_polygon_l3967_396771

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define a triangle as a set of three points in 2D space
def Triangle : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Function to check if a triangle is inside a polygon
def isInside (t : Triangle) (p : Polygon) : Prop := sorry

-- Function to calculate the perimeter of a polygon
def perimeterPolygon (p : Polygon) : ℝ := sorry

-- Function to calculate the perimeter of a triangle
def perimeterTriangle (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_perimeter_inside_polygon (t : Triangle) (p : Polygon) :
  isInside t p → perimeterTriangle t ≤ perimeterPolygon p := by sorry

end triangle_perimeter_inside_polygon_l3967_396771


namespace melanie_gumball_sale_l3967_396728

/-- Represents the sale of gumballs -/
structure GumballSale where
  price_per_gumball : ℕ
  total_money : ℕ

/-- Calculates the number of gumballs sold -/
def gumballs_sold (sale : GumballSale) : ℕ :=
  sale.total_money / sale.price_per_gumball

/-- Theorem: Melanie sold 4 gumballs -/
theorem melanie_gumball_sale :
  let sale : GumballSale := { price_per_gumball := 8, total_money := 32 }
  gumballs_sold sale = 4 := by
  sorry

end melanie_gumball_sale_l3967_396728


namespace p_and_q_true_l3967_396793

theorem p_and_q_true : 
  (∃ x₀ : ℝ, Real.tan x₀ = Real.sqrt 3) ∧ 
  (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end p_and_q_true_l3967_396793


namespace max_min_sum_zero_l3967_396794

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x, f x = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x, f x = n) ∧
               (m + n = 0) := by
  sorry

end max_min_sum_zero_l3967_396794


namespace solution_set_quadratic_inequality_l3967_396769

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x^2 + 5*x - 24 < 0 ↔ -8 < x ∧ x < 3 := by
  sorry

end solution_set_quadratic_inequality_l3967_396769


namespace arc_length_parametric_curve_l3967_396790

open Real MeasureTheory

/-- The arc length of the curve given by the parametric equations
    x = e^t(cos t + sin t) and y = e^t(cos t - sin t) for 0 ≤ t ≤ 2π -/
theorem arc_length_parametric_curve :
  let x : ℝ → ℝ := fun t ↦ exp t * (cos t + sin t)
  let y : ℝ → ℝ := fun t ↦ exp t * (cos t - sin t)
  let curve_length := ∫ t in Set.Icc 0 (2 * π), sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)
  curve_length = 2 * (exp (2 * π) - 1) := by
sorry

end arc_length_parametric_curve_l3967_396790


namespace min_sum_theorem_l3967_396744

-- Define the equation
def equation (x y : ℝ) : Prop := -x^2 + 7*x + y - 10 = 0

-- Define the sum function
def sum (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem min_sum_theorem :
  ∃ (min : ℝ), min = 1 ∧ 
  (∀ x y : ℝ, equation x y → sum x y ≥ min) ∧
  (∃ x y : ℝ, equation x y ∧ sum x y = min) :=
sorry

end min_sum_theorem_l3967_396744


namespace divisibility_condition_l3967_396737

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2*p) :
  (n^(p-1) ∣ ((p-1)^n + 1)) ↔ (n = 1 ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end divisibility_condition_l3967_396737


namespace total_pages_in_book_l3967_396788

theorem total_pages_in_book (pages_read : ℕ) (pages_left : ℕ) : 
  pages_read = 147 → pages_left = 416 → pages_read + pages_left = 563 := by
sorry

end total_pages_in_book_l3967_396788


namespace probability_red_or_white_marble_l3967_396782

theorem probability_red_or_white_marble (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h1 : total = 60) 
  (h2 : blue = 5) 
  (h3 : red = 9) :
  (red : ℚ) / total + ((total - blue - red) : ℚ) / total = 11 / 12 := by
  sorry

end probability_red_or_white_marble_l3967_396782


namespace store_sales_increase_l3967_396754

/-- Proves that if a store's sales increased by 25% to $400 million, then the previous year's sales were $320 million. -/
theorem store_sales_increase (current_sales : ℝ) (increase_percent : ℝ) :
  current_sales = 400 ∧ increase_percent = 0.25 →
  (1 + increase_percent) * (current_sales / (1 + increase_percent)) = 320 := by
  sorry

end store_sales_increase_l3967_396754


namespace intersection_of_A_and_B_l3967_396780

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l3967_396780


namespace rectangle_area_l3967_396726

/-- The area of a rectangle with sides of length (a - b) and (c + d) is equal to ac + ad - bc - bd. -/
theorem rectangle_area (a b c d : ℝ) : 
  let length := a - b
  let width := c + d
  length * width = a*c + a*d - b*c - b*d := by sorry

end rectangle_area_l3967_396726


namespace min_quadrilateral_area_l3967_396724

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The ellipse passes through the point (1, √2/2) -/
axiom point_on_ellipse : ellipse 1 (Real.sqrt 2 / 2)

/-- The point (1,0) is a focus of the ellipse -/
axiom focus_point : ∃ c, c^2 = 1 ∧ c^2 = 2 - 1

/-- Definition of perpendicular lines through (1,0) -/
def perpendicular_lines (m₁ m₂ : ℝ) : Prop := 
  m₁ * m₂ = -1 ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- Definition of the area of the quadrilateral formed by intersection points -/
noncomputable def quadrilateral_area (m₁ m₂ : ℝ) : ℝ := 
  4 * (m₁^2 + 1)^2 / ((m₁^2 + 2) * (2 * m₂^2 + 1))

/-- The main theorem to prove -/
theorem min_quadrilateral_area : 
  ∃ (m₁ m₂ : ℝ), perpendicular_lines m₁ m₂ ∧ 
  (∀ (n₁ n₂ : ℝ), perpendicular_lines n₁ n₂ → 
    quadrilateral_area m₁ m₂ ≤ quadrilateral_area n₁ n₂) ∧
  quadrilateral_area m₁ m₂ = 16/9 :=
sorry

end min_quadrilateral_area_l3967_396724


namespace sin_eq_cos_condition_l3967_396715

open Real

theorem sin_eq_cos_condition (α : ℝ) :
  (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α ∧
  ¬ (sin α = cos α → ∃ k : ℤ, α = π / 4 + 2 * k * π) :=
by sorry

end sin_eq_cos_condition_l3967_396715


namespace range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l3967_396730

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

-- Theorem 1: Range of x when f(x) ≤ 1
theorem range_of_x_when_f_leq_1 :
  {x : ℝ | f x ≤ 1} = Set.Icc 0 6 := by sorry

-- Theorem 2: Range of m when f(x) - g(x) ≥ m+1 for all x ∈ ℝ
theorem range_of_m_when_f_minus_g_geq_m_plus_1 :
  {m : ℝ | ∀ x, f x - g x ≥ m + 1} = Set.Iic (-3) := by sorry

end range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l3967_396730


namespace cottage_rental_cost_l3967_396784

theorem cottage_rental_cost (cost_per_hour : ℝ) (rental_hours : ℝ) (num_friends : ℕ) :
  cost_per_hour = 5 →
  rental_hours = 8 →
  num_friends = 2 →
  (cost_per_hour * rental_hours) / num_friends = 20 := by
  sorry

end cottage_rental_cost_l3967_396784


namespace total_is_245_l3967_396760

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem setup -/
def problem_setup (d : MoneyDistribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.30 * d.x ∧ d.y = 63

/-- The total amount distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.x + d.y + d.z

/-- The theorem stating the total amount is 245 rupees -/
theorem total_is_245 (d : MoneyDistribution) (h : problem_setup d) : total_amount d = 245 := by
  sorry


end total_is_245_l3967_396760


namespace card_drawing_theorem_l3967_396798

/-- The number of cards of each color -/
def cards_per_color : ℕ := 4

/-- The total number of cards -/
def total_cards : ℕ := 4 * cards_per_color

/-- The number of cards to be drawn -/
def cards_drawn : ℕ := 3

/-- The number of ways to draw cards satisfying the conditions -/
def valid_draws : ℕ := 472

theorem card_drawing_theorem : 
  (Nat.choose total_cards cards_drawn) - 
  (4 * Nat.choose cards_per_color cards_drawn) - 
  (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) = 
  valid_draws := by sorry

end card_drawing_theorem_l3967_396798


namespace stratified_sample_size_l3967_396739

/-- Calculates the sample size for a stratified sampling based on gender -/
def calculateSampleSize (totalEmployees : ℕ) (maleEmployees : ℕ) (maleSampleSize : ℕ) : ℕ :=
  (totalEmployees * maleSampleSize) / maleEmployees

/-- Proves that the sample size is 24 given the conditions -/
theorem stratified_sample_size :
  let totalEmployees : ℕ := 120
  let maleEmployees : ℕ := 90
  let maleSampleSize : ℕ := 18
  calculateSampleSize totalEmployees maleEmployees maleSampleSize = 24 := by
  sorry

end stratified_sample_size_l3967_396739


namespace green_ducks_percentage_l3967_396700

theorem green_ducks_percentage (smaller_pond : ℕ) (larger_pond : ℕ) 
  (larger_pond_green_percent : ℝ) (total_green_percent : ℝ) :
  smaller_pond = 30 →
  larger_pond = 50 →
  larger_pond_green_percent = 12 →
  total_green_percent = 15 →
  (smaller_pond_green_percent : ℝ) * smaller_pond / 100 + 
    larger_pond_green_percent * larger_pond / 100 = 
    total_green_percent * (smaller_pond + larger_pond) / 100 →
  smaller_pond_green_percent = 20 :=
by
  sorry

end green_ducks_percentage_l3967_396700


namespace fence_cost_calculation_l3967_396792

def parallel_side1_length : ℕ := 25
def parallel_side2_length : ℕ := 37
def non_parallel_side1_length : ℕ := 20
def non_parallel_side2_length : ℕ := 24
def parallel_side_price : ℕ := 48
def non_parallel_side_price : ℕ := 60

theorem fence_cost_calculation :
  (parallel_side1_length * parallel_side_price) +
  (parallel_side2_length * parallel_side_price) +
  (non_parallel_side1_length * non_parallel_side_price) +
  (non_parallel_side2_length * non_parallel_side_price) = 5616 := by
  sorry

end fence_cost_calculation_l3967_396792


namespace composite_for_n_greater_than_two_l3967_396742

def number_with_ones_and_seven (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

theorem composite_for_n_greater_than_two :
  ∀ n : ℕ, n > 2 → ¬(Nat.Prime (number_with_ones_and_seven n)) :=
sorry

end composite_for_n_greater_than_two_l3967_396742


namespace oblique_asymptote_of_f_l3967_396741

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 5) / (x + 4)

theorem oblique_asymptote_of_f :
  ∃ (a b : ℝ), a ≠ 0 ∧ (∀ ε > 0, ∃ M, ∀ x > M, |f x - (a * x + b)| < ε) ∧ a = 3 ∧ b = -4 :=
sorry

end oblique_asymptote_of_f_l3967_396741


namespace events_mutually_exclusive_not_complementary_l3967_396727

/-- Represents the number of boys in the group -/
def num_boys : Nat := 3

/-- Represents the number of girls in the group -/
def num_girls : Nat := 2

/-- Represents the number of students to be selected -/
def num_selected : Nat := 2

/-- Event: Exactly 1 boy is selected -/
def event_one_boy (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val < num_boys)).card = 1

/-- Event: Exactly 2 girls are selected -/
def event_two_girls (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val ≥ num_boys)).card = 2

/-- Theorem: The events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (∀ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬(event_one_boy selected ∧ event_two_girls selected)) ∧
  (∃ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬event_one_boy selected ∧ ¬event_two_girls selected) :=
sorry

end events_mutually_exclusive_not_complementary_l3967_396727


namespace buqing_college_students_l3967_396729

/-- Represents the number of students in each college -/
structure CollegeStudents where
  a₁ : ℕ  -- Buqing College
  a₂ : ℕ  -- Jiazhen College
  a₃ : ℕ  -- Hede College
  a₄ : ℕ  -- Wangdao College

/-- Checks if the given numbers form an arithmetic sequence with the specified common difference -/
def isArithmeticSequence (a b c : ℕ) (d : ℕ) : Prop :=
  b = a + d ∧ c = b + d

/-- Checks if the given numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

/-- The main theorem to prove -/
theorem buqing_college_students 
  (s : CollegeStudents) 
  (total : s.a₁ + s.a₂ + s.a₃ + s.a₄ = 474) 
  (arith_seq : isArithmeticSequence s.a₁ s.a₂ s.a₃ 12)
  (geom_seq : isGeometricSequence s.a₁ s.a₃ s.a₄) : 
  s.a₁ = 96 := by
  sorry

end buqing_college_students_l3967_396729


namespace complex_sum_problem_l3967_396736

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 2 → 
  w = -x - u → 
  Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.mk 2 (-1) → 
  v + z = -3 := by sorry

end complex_sum_problem_l3967_396736


namespace quadratic_constraint_l3967_396777

/-- Quadratic function defined by a parameter a -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

/-- Theorem stating the condition on a for the given constraints -/
theorem quadratic_constraint (a : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : a ≠ 0)
  (h₂ : x₁ + x₂ = 2)
  (h₃ : x₁ < x₂)
  (h₄ : y₁ > y₂)
  (h₅ : quadratic_function a x₁ = y₁)
  (h₆ : quadratic_function a x₂ = y₂) :
  a < -2/5 := by
  sorry

end quadratic_constraint_l3967_396777


namespace factorial_simplification_l3967_396761

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end factorial_simplification_l3967_396761


namespace right_triangle_area_l3967_396799

theorem right_triangle_area (h : ℝ) (h_pos : h > 0) : ∃ (a b : ℝ),
  a > 0 ∧ b > 0 ∧
  a / b = 3 / 4 ∧
  a^2 + b^2 = h^2 ∧
  (1/2) * a * b = (6/25) * h^2 := by
sorry

end right_triangle_area_l3967_396799


namespace smallest_solution_floor_equation_l3967_396734

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 109 ∧
  (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 19 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 19 :=
sorry

end smallest_solution_floor_equation_l3967_396734


namespace product_of_twelve_and_3460_l3967_396722

theorem product_of_twelve_and_3460 : ∃ x : ℕ, 12 * x = 173 * x ∧ x = 3460 → 12 * 3460 = 41520 := by
  sorry

end product_of_twelve_and_3460_l3967_396722


namespace hotdog_problem_l3967_396723

theorem hotdog_problem (h1 h2 : ℕ) : 
  h2 = h1 - 25 → 
  h1 + h2 = 125 → 
  h1 = 75 := by
sorry

end hotdog_problem_l3967_396723


namespace M_subset_P_l3967_396703

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}
def P : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2)}

theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l3967_396703


namespace sufficient_not_necessary_l3967_396765

theorem sufficient_not_necessary : 
  (∀ x : ℝ, (|x| < 2 → x^2 - x - 6 < 0)) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end sufficient_not_necessary_l3967_396765


namespace abs_inequality_l3967_396762

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end abs_inequality_l3967_396762


namespace inverse_variation_problem_l3967_396787

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^2 = k) →  -- a^3 varies inversely with b^2
  (4^3 * 2^2 = k) →         -- a = 4 when b = 2
  (a^3 * 8^2 = k) →         -- condition for b = 8
  a = 4^(1/3) :=            -- prove that a = 4^(1/3) when b = 8
by sorry

end inverse_variation_problem_l3967_396787


namespace two_common_tangents_range_l3967_396750

/-- Two circles in a 2D plane --/
structure TwoCircles where
  a : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 2)^2 + y^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + (y + 3)^2 = 9

/-- The condition for two circles to have exactly two common tangents --/
def has_two_common_tangents (circles : TwoCircles) : Prop :=
  1 < Real.sqrt ((circles.a - 2)^2 + 9) ∧ Real.sqrt ((circles.a - 2)^2 + 9) < 5

/-- Theorem stating the range of 'a' for which the circles have exactly two common tangents --/
theorem two_common_tangents_range (circles : TwoCircles) :
  has_two_common_tangents circles ↔ -2 < circles.a ∧ circles.a < 6 := by
  sorry


end two_common_tangents_range_l3967_396750


namespace economic_formula_solution_l3967_396712

theorem economic_formula_solution (p x : ℂ) :
  (3 * p - x = 15000) → (x = 9 + 225 * Complex.I) → (p = 5003 + 75 * Complex.I) :=
by sorry

end economic_formula_solution_l3967_396712


namespace factor_of_polynomial_l3967_396776

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 - 6*x^2 + 9 : ℝ) = (x^2 - 3) * q x := by
  sorry

end factor_of_polynomial_l3967_396776


namespace sasha_can_get_123_l3967_396711

/-- Represents an arithmetic expression --/
inductive Expr
  | Num : Nat → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression --/
def eval : Expr → Int
  | Expr.Num n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each number from 1 to 5 exactly once --/
def usesAllNumbers : Expr → Bool := sorry

theorem sasha_can_get_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end sasha_can_get_123_l3967_396711


namespace rectangle_width_l3967_396753

theorem rectangle_width (square_area : ℝ) (rectangle_length : ℝ) (square_perimeter : ℝ) :
  square_area = 5 * (rectangle_length * 10) ∧
  rectangle_length = 50 ∧
  square_perimeter = 200 ∧
  square_area = (square_perimeter / 4) ^ 2 →
  10 = 10 := by
sorry

end rectangle_width_l3967_396753


namespace find_m_l3967_396709

theorem find_m : ∃ m : ℤ, 3^4 - 6 = 5^2 + m ∧ m = 50 := by
  sorry

end find_m_l3967_396709
