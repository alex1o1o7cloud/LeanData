import Mathlib

namespace hexagonal_circle_selection_l1818_181846

/-- Represents the number of ways to choose three consecutive circles in a direction --/
def consecutive_triples (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of circles in the figure --/
def total_circles : ℕ := 33

/-- The number of circles in the longest row --/
def longest_row : ℕ := 6

/-- The number of ways to choose three consecutive circles in the first direction --/
def first_direction : ℕ := consecutive_triples longest_row

/-- The number of ways to choose three consecutive circles in each of the other two directions --/
def other_directions : ℕ := 18

/-- The total number of ways to choose three consecutive circles in all directions --/
def total_ways : ℕ := first_direction + 2 * other_directions

theorem hexagonal_circle_selection :
  total_ways = 57 :=
sorry

end hexagonal_circle_selection_l1818_181846


namespace parallelogram_area_l1818_181810

/-- The area of a parallelogram is the product of two adjacent sides and the sine of the angle between them. -/
theorem parallelogram_area (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃ (S : ℝ), S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end parallelogram_area_l1818_181810


namespace sum_of_digits_power_product_l1818_181856

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_power_product :
  sumOfDigits (2^2010 * 5^2012 * 7) = 13 := by
  sorry

end sum_of_digits_power_product_l1818_181856


namespace jacket_cost_calculation_l1818_181818

/-- The amount spent on clothes in cents -/
def total_spent : ℕ := 1428

/-- The amount spent on shorts in cents -/
def shorts_cost : ℕ := 954

/-- The amount spent on the jacket in cents -/
def jacket_cost : ℕ := total_spent - shorts_cost

theorem jacket_cost_calculation : jacket_cost = 474 := by
  sorry

end jacket_cost_calculation_l1818_181818


namespace laptop_final_price_l1818_181887

/-- The final price of a laptop after successive discounts --/
theorem laptop_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) :
  original_price = 1200 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 820.80 := by
  sorry

#check laptop_final_price

end laptop_final_price_l1818_181887


namespace equivalent_discount_l1818_181819

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount1 = 0.3)
  (h3 : discount2 = 0.2) : 
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.44) :=
by sorry

end equivalent_discount_l1818_181819


namespace fraction_sum_and_product_l1818_181821

theorem fraction_sum_and_product : 
  (2 / 16 + 3 / 18 + 4 / 24) * (3 / 5) = 11 / 40 := by
  sorry

end fraction_sum_and_product_l1818_181821


namespace quadratic_roots_l1818_181854

theorem quadratic_roots (p q a b : ℤ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →  -- polynomial has roots a and b
  a ≠ b →                                    -- roots are distinct
  a ≠ 0 →                                    -- a is non-zero
  b ≠ 0 →                                    -- b is non-zero
  (a + p) % (q - 2*b) = 0 →                  -- a + p is divisible by q - 2b
  a = 1 ∨ a = 3 :=                           -- possible values for a
by sorry

end quadratic_roots_l1818_181854


namespace lcm_product_geq_lcm_square_l1818_181861

theorem lcm_product_geq_lcm_square (k m n : ℕ) :
  Nat.lcm (Nat.lcm k m) n * Nat.lcm (Nat.lcm m n) k * Nat.lcm (Nat.lcm n k) m ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
  sorry

end lcm_product_geq_lcm_square_l1818_181861


namespace song_listens_after_three_months_l1818_181862

/-- Calculates the total listens for a song that doubles in popularity each month -/
def totalListens (initialListens : ℕ) (months : ℕ) : ℕ :=
  let doublingSequence := List.range months |>.map (fun i => initialListens * 2^(i + 1))
  initialListens + doublingSequence.sum

/-- Theorem: The total listens after 3 months of doubling is 900,000 given 60,000 initial listens -/
theorem song_listens_after_three_months :
  totalListens 60000 3 = 900000 := by
  sorry

end song_listens_after_three_months_l1818_181862


namespace isabel_homework_problems_l1818_181841

/-- The total number of problems Isabel has to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
  (math_problems_per_page reading_problems_per_page : ℕ)
  (science_problems_per_page history_problems : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems

/-- Theorem stating that Isabel has to complete 61 problems in total -/
theorem isabel_homework_problems :
  total_problems 2 4 3 1 5 5 7 10 = 61 := by
  sorry

end isabel_homework_problems_l1818_181841


namespace f_properties_l1818_181884

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f (x + T) = f x) ∧  -- f has period T
    T = 2 * Real.pi ∧  -- The period is 2π
    (∀ x, f x ≤ max_value) ∧  -- max_value is an upper bound
    max_value = Real.sqrt 2 ∧  -- The maximum value is √2
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧  -- max_set contains all x where f(x) is maximum
    (∀ k : ℤ, (2 * k : ℝ) * Real.pi + 3 * Real.pi / 4 ∈ max_set)  -- Characterization of max_set
    := by sorry

end f_properties_l1818_181884


namespace sue_movie_borrowing_l1818_181820

/-- The number of movies Sue initially borrowed -/
def initial_movies : ℕ := 6

/-- The number of books Sue initially borrowed -/
def initial_books : ℕ := 15

/-- The number of books Sue returned -/
def returned_books : ℕ := 8

/-- The number of additional books Sue checked out -/
def additional_books : ℕ := 9

/-- The total number of items Sue has at the end -/
def total_items : ℕ := 20

theorem sue_movie_borrowing :
  initial_movies = 6 ∧
  initial_books + initial_movies - returned_books - (initial_movies / 3) + additional_books = total_items :=
by sorry

end sue_movie_borrowing_l1818_181820


namespace original_price_correct_l1818_181828

/-- The original price of a single article before discounts and taxes -/
def original_price : ℝ := 669.99

/-- The discount rate for purchases of 2 or more articles -/
def discount_rate : ℝ := 0.24

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- The number of articles purchased -/
def num_articles : ℕ := 3

/-- The total cost after discount and tax -/
def total_cost : ℝ := 1649.43

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  num_articles * (original_price * (1 - discount_rate)) * (1 + sales_tax_rate) = total_cost := by
  sorry

end original_price_correct_l1818_181828


namespace evaluate_expression_l1818_181882

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/3) 
  (hz : z = 12) 
  (hw : w = -2) : 
  x^2 * y^3 * z * w = -1/18 := by
  sorry

end evaluate_expression_l1818_181882


namespace percentage_of_juniors_l1818_181800

def total_students : ℕ := 800
def seniors : ℕ := 160

theorem percentage_of_juniors : 
  ∀ (freshmen sophomores juniors : ℕ),
  freshmen + sophomores + juniors + seniors = total_students →
  sophomores = total_students / 4 →
  freshmen = sophomores + 32 →
  (juniors : ℚ) / total_students * 100 = 26 :=
by sorry

end percentage_of_juniors_l1818_181800


namespace intersection_equals_T_l1818_181802

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_equals_T : S ∩ T = T := by sorry

end intersection_equals_T_l1818_181802


namespace inverse_proportion_example_l1818_181830

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 1 = 40 →
  y 1 = 8 →
  y 2 = 20 →
  x 2 = 16 := by
sorry

end inverse_proportion_example_l1818_181830


namespace equation_solutions_l1818_181837

def solution_set : Set (ℤ × ℤ) :=
  {(6, 3), (6, -9), (1, 1), (1, -2), (2, -1)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  y * (x + y) = x^3 - 7*x^2 + 11*x - 3

theorem equation_solutions :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set :=
sorry

end equation_solutions_l1818_181837


namespace sexagenary_cycle_2016_2017_l1818_181896

/-- Represents the Heavenly Stems in the Sexagenary cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Returns the next Heavenly Stem in the cycle -/
def nextStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Yi
  | HeavenlyStem.Yi => HeavenlyStem.Bing
  | HeavenlyStem.Bing => HeavenlyStem.Ding
  | HeavenlyStem.Ding => HeavenlyStem.Wu
  | HeavenlyStem.Wu => HeavenlyStem.Ji
  | HeavenlyStem.Ji => HeavenlyStem.Geng
  | HeavenlyStem.Geng => HeavenlyStem.Xin
  | HeavenlyStem.Xin => HeavenlyStem.Ren
  | HeavenlyStem.Ren => HeavenlyStem.Gui
  | HeavenlyStem.Gui => HeavenlyStem.Jia

/-- Returns the next Earthly Branch in the cycle -/
def nextBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Chou
  | EarthlyBranch.Chou => EarthlyBranch.Yin
  | EarthlyBranch.Yin => EarthlyBranch.Mao
  | EarthlyBranch.Mao => EarthlyBranch.Chen
  | EarthlyBranch.Chen => EarthlyBranch.Si
  | EarthlyBranch.Si => EarthlyBranch.Wu
  | EarthlyBranch.Wu => EarthlyBranch.Wei
  | EarthlyBranch.Wei => EarthlyBranch.Shen
  | EarthlyBranch.Shen => EarthlyBranch.You
  | EarthlyBranch.You => EarthlyBranch.Xu
  | EarthlyBranch.Xu => EarthlyBranch.Hai
  | EarthlyBranch.Hai => EarthlyBranch.Zi

/-- Returns the next year in the Sexagenary cycle -/
def nextYear (y : SexagenaryYear) : SexagenaryYear :=
  { stem := nextStem y.stem, branch := nextBranch y.branch }

theorem sexagenary_cycle_2016_2017 :
  ∀ (y2016 : SexagenaryYear),
    y2016.stem = HeavenlyStem.Bing ∧ y2016.branch = EarthlyBranch.Shen →
    (nextYear y2016).stem = HeavenlyStem.Ding ∧ (nextYear y2016).branch = EarthlyBranch.You :=
by sorry

end sexagenary_cycle_2016_2017_l1818_181896


namespace roger_step_goal_time_l1818_181855

/-- Represents the time it takes Roger to reach his step goal -/
def time_to_reach_goal (steps_per_interval : ℕ) (interval_duration : ℕ) (goal_steps : ℕ) : ℕ :=
  (goal_steps * interval_duration) / steps_per_interval

/-- Proves that Roger will take 150 minutes to reach his goal of 10,000 steps -/
theorem roger_step_goal_time :
  time_to_reach_goal 2000 30 10000 = 150 := by
  sorry

end roger_step_goal_time_l1818_181855


namespace complement_N_intersect_M_l1818_181891

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Define set N
def N : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_N_intersect_M :
  (U \ N) ∩ M = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end complement_N_intersect_M_l1818_181891


namespace arithmetic_sequence_sum_l1818_181833

/-- An arithmetic sequence with first term -2015 -/
def arithmetic_sequence (n : ℕ) : ℤ := -2015 + (n - 1) * d
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (2 * (-2015) + (n - 1) * d) / 2
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Main theorem -/
theorem arithmetic_sequence_sum :
  2 * S 6 - 3 * S 4 = 24 → S 2015 = -2015 := by
  sorry

end arithmetic_sequence_sum_l1818_181833


namespace pistachios_with_opened_shells_l1818_181879

/-- Given a bag of pistachios, calculate the number of pistachios with shells and opened shells -/
theorem pistachios_with_opened_shells
  (total : ℕ)
  (shell_percent : ℚ)
  (opened_percent : ℚ)
  (h_total : total = 80)
  (h_shell : shell_percent = 95 / 100)
  (h_opened : opened_percent = 75 / 100) :
  ⌊(total : ℚ) * shell_percent * opened_percent⌋ = 57 := by
sorry

end pistachios_with_opened_shells_l1818_181879


namespace ellipse_equation_l1818_181878

/-- An ellipse passing through (-√15, 5/2) with the same foci as 9x^2 + 4y^2 = 36 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = b^2 + 5 ∧ 
   x^2 / b^2 + y^2 / a^2 = 1 ∧
   (-Real.sqrt 15)^2 / b^2 + (5/2)^2 / a^2 = 1) →
  x^2 / 20 + y^2 / 25 = 1 :=
sorry

end ellipse_equation_l1818_181878


namespace derivative_f_at_pi_fourth_l1818_181869

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 1

theorem derivative_f_at_pi_fourth : 
  (deriv f) (π/4) = Real.sqrt 2 := by sorry

end derivative_f_at_pi_fourth_l1818_181869


namespace dagger_example_l1818_181877

-- Define the ternary operation ⋄
def dagger (a b c d e f : ℚ) : ℚ := (a * c * e) * ((d * f) / b)

-- Theorem statement
theorem dagger_example : dagger 5 9 7 2 11 5 = 3850 / 9 := by
  sorry

end dagger_example_l1818_181877


namespace solution_set_f_geq_3_range_of_a_l1818_181809

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ∈ Set.Icc (-2) 3 := by sorry


end solution_set_f_geq_3_range_of_a_l1818_181809


namespace parabola_properties_l1818_181883

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Theorem statement
theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x y : ℝ, f ((x + y) / 2) ≤ (f x + f y) / 2) ∧
  -- 2. The axis of symmetry is x = 2
  (∀ h : ℝ, f (2 + h) = f (2 - h)) ∧
  -- 3. The vertex is at (2, 1)
  (f 2 = 1 ∧ ∀ x : ℝ, f x ≥ 1) ∧
  -- 4. When x < 2, y decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) :=
by
  sorry

end parabola_properties_l1818_181883


namespace parallel_vectors_subtraction_l1818_181876

/-- Given two parallel 2D vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end parallel_vectors_subtraction_l1818_181876


namespace sqrt_fraction_equals_two_power_fifteen_l1818_181839

theorem sqrt_fraction_equals_two_power_fifteen :
  let thirty_two := (2 : ℝ) ^ 5
  let sixteen := (2 : ℝ) ^ 4
  (((thirty_two ^ 15 + sixteen ^ 15) / (thirty_two ^ 6 + sixteen ^ 18)) ^ (1/2 : ℝ)) = (2 : ℝ) ^ 15 := by
  sorry

end sqrt_fraction_equals_two_power_fifteen_l1818_181839


namespace babysitting_age_ratio_l1818_181863

theorem babysitting_age_ratio : 
  ∀ (jane_start_age jane_current_age jane_stop_years_ago oldest_babysat_current_age : ℕ),
    jane_start_age = 16 →
    jane_current_age = 32 →
    jane_stop_years_ago = 10 →
    oldest_babysat_current_age = 24 →
    ∃ (child_age jane_age : ℕ),
      child_age = oldest_babysat_current_age - jane_stop_years_ago ∧
      jane_age = jane_current_age - jane_stop_years_ago ∧
      child_age * 11 = jane_age * 7 := by
  sorry

end babysitting_age_ratio_l1818_181863


namespace vector_subtraction_l1818_181811

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end vector_subtraction_l1818_181811


namespace polynomial_factorization_l1818_181888

theorem polynomial_factorization (x : ℝ) :
  (x^3 - x + 3)^2 = x^6 - 2*x^4 + 6*x^3 + x^2 - 6*x + 9 := by
  sorry

end polynomial_factorization_l1818_181888


namespace range_of_a_l1818_181842

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ (a ≤ 0 ∨ a ≥ 6) :=
by sorry

end range_of_a_l1818_181842


namespace fraction_product_simplification_l1818_181875

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l1818_181875


namespace parabola_translation_l1818_181870

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 6 0 0
  let translated := translate original 2 3
  y = 6 * x^2 → y = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c :=
by sorry

end parabola_translation_l1818_181870


namespace catch_in_park_l1818_181822

-- Define the square park
structure Park :=
  (side_length : ℝ)
  (has_diagonal_walkways : Bool)

-- Define the participants
structure Participant :=
  (speed : ℝ)
  (position : ℝ × ℝ)

-- Define the catching condition
def can_catch (pursuer1 pursuer2 target : Participant) (park : Park) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer1.position = target.position ∨ pursuer2.position = target.position)

-- Theorem statement
theorem catch_in_park (park : Park) (pursuer1 pursuer2 target : Participant) :
  park.side_length > 0 ∧
  park.has_diagonal_walkways = true ∧
  pursuer1.speed > 0 ∧
  pursuer2.speed > 0 ∧
  target.speed = 3 * pursuer1.speed ∧
  target.speed = 3 * pursuer2.speed →
  can_catch pursuer1 pursuer2 target park :=
sorry

end catch_in_park_l1818_181822


namespace square_root_of_four_l1818_181812

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l1818_181812


namespace pet_store_birds_l1818_181898

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 40 :=
sorry

end pet_store_birds_l1818_181898


namespace bread_price_calculation_bread_price_proof_l1818_181817

theorem bread_price_calculation (initial_price : ℝ) 
  (thursday_increase : ℝ) (saturday_discount : ℝ) : ℝ :=
  let thursday_price := initial_price * (1 + thursday_increase)
  let saturday_price := thursday_price * (1 - saturday_discount)
  saturday_price

theorem bread_price_proof :
  bread_price_calculation 50 0.2 0.15 = 51 := by
  sorry

end bread_price_calculation_bread_price_proof_l1818_181817


namespace dividend_calculation_l1818_181801

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end dividend_calculation_l1818_181801


namespace calculate_savings_l1818_181804

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 18000 ∧ income_ratio = 5 ∧ expenditure_ratio = 4 →
  income - (income * expenditure_ratio / income_ratio) = 3600 := by
sorry

end calculate_savings_l1818_181804


namespace negation_of_forall_positive_negation_of_original_statement_l1818_181873

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_original_statement :
  (¬ ∀ x > 0, x^2 - 3*x + 2 < 0) ↔ (∃ x > 0, x^2 - 3*x + 2 ≥ 0) :=
by sorry

end negation_of_forall_positive_negation_of_original_statement_l1818_181873


namespace local_min_implies_a_eq_one_l1818_181881

/-- The function f(x) = x³ - 2ax² + a²x + 1 has a local minimum at x = 1 -/
def has_local_min_at_one (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x ≥ f 1

/-- The function f(x) = x³ - 2ax² + a²x + 1 -/
def f (a x : ℝ) : ℝ := x^3 - 2*a*x^2 + a^2*x + 1

theorem local_min_implies_a_eq_one (a : ℝ) :
  has_local_min_at_one (f a) a → a = 1 := by
  sorry

end local_min_implies_a_eq_one_l1818_181881


namespace square_nine_on_top_l1818_181871

-- Define the grid of squares
def Grid := Fin 4 → Fin 4 → Fin 16

-- Define the initial configuration of the grid
def initial_grid : Grid :=
  fun i j => i * 4 + j + 1

-- Define the folding operations
def fold_top_over_bottom (g : Grid) : Grid :=
  fun i j => g (3 - i) j

def fold_bottom_over_top (g : Grid) : Grid :=
  fun i j => g i j

def fold_right_over_left (g : Grid) : Grid :=
  fun i j => g i (3 - j)

def fold_left_over_right (g : Grid) : Grid :=
  fun i j => g i j

-- Define the complete folding sequence
def fold_sequence (g : Grid) : Grid :=
  fold_left_over_right ∘ fold_right_over_left ∘ fold_bottom_over_top ∘ fold_top_over_bottom $ g

-- Theorem stating that after the folding sequence, square 9 is on top
theorem square_nine_on_top :
  (fold_sequence initial_grid) 0 0 = 9 := by
  sorry

end square_nine_on_top_l1818_181871


namespace categorical_variables_are_correct_l1818_181814

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | Religious_Belief
  | Nationality

-- Define a function to check if a variable is categorical
def is_categorical (v : Variable) : Prop :=
  v = Variable.Gender ∨ v = Variable.Religious_Belief ∨ v = Variable.Nationality

-- Define the set of all variables
def all_variables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.Religious_Belief, Variable.Nationality}

-- Define the set of categorical variables
def categorical_variables : Set Variable :=
  {v ∈ all_variables | is_categorical v}

-- The theorem to prove
theorem categorical_variables_are_correct :
  categorical_variables = {Variable.Gender, Variable.Religious_Belief, Variable.Nationality} :=
by sorry

end categorical_variables_are_correct_l1818_181814


namespace race_time_proof_l1818_181823

/-- Represents the race times of two runners -/
structure RaceTimes where
  total : ℕ
  difference : ℕ

/-- Calculates the longer race time given the total time and the difference between runners -/
def longerTime (times : RaceTimes) : ℕ :=
  (times.total + times.difference) / 2

theorem race_time_proof (times : RaceTimes) (h1 : times.total = 112) (h2 : times.difference = 4) :
  longerTime times = 58 := by
  sorry

end race_time_proof_l1818_181823


namespace power_difference_equality_l1818_181890

theorem power_difference_equality : 2^2014 - (-2)^2015 = 3 * 2^2014 := by
  sorry

end power_difference_equality_l1818_181890


namespace arithmetic_sequence_sum_l1818_181885

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a → a 5 = 3 → a 6 = -2 →
  (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) = 3 :=
by
  sorry

end arithmetic_sequence_sum_l1818_181885


namespace leftHandedLikeMusicalCount_l1818_181806

/-- Represents the Drama Club -/
structure DramaClub where
  total : Nat
  leftHanded : Nat
  likeMusical : Nat
  rightHandedDislike : Nat

/-- The number of left-handed people who like musical theater in the Drama Club -/
def leftHandedLikeMusical (club : DramaClub) : Nat :=
  club.leftHanded + club.likeMusical - (club.total - club.rightHandedDislike)

/-- Theorem stating the number of left-handed musical theater lovers in the specific Drama Club -/
theorem leftHandedLikeMusicalCount : leftHandedLikeMusical { 
  total := 25,
  leftHanded := 10,
  likeMusical := 18,
  rightHandedDislike := 3
} = 6 := by sorry

end leftHandedLikeMusicalCount_l1818_181806


namespace simplify_expression_l1818_181872

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x^2 + y^2 + z^2 = x*y + y*z + z*x) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 / x^2 := by
  sorry

end simplify_expression_l1818_181872


namespace p_greater_than_q_l1818_181853

theorem p_greater_than_q (x y : ℝ) (h1 : x < y) (h2 : y < 0) 
  (p : ℝ := (x^2 + y^2)*(x - y)) (q : ℝ := (x^2 - y^2)*(x + y)) : p > q := by
  sorry

end p_greater_than_q_l1818_181853


namespace exceed_permutations_l1818_181865

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 3

theorem exceed_permutations :
  factorial word_length / factorial repeated_letter_count = 120 := by
  sorry

end exceed_permutations_l1818_181865


namespace train_journey_duration_l1818_181848

/-- Given a train journey with a distance and average speed, calculate the duration of the journey. -/
theorem train_journey_duration (distance : ℝ) (speed : ℝ) (duration : ℝ) 
  (h_distance : distance = 27) 
  (h_speed : speed = 3) 
  (h_duration : duration = distance / speed) : 
  duration = 9 := by
  sorry

end train_journey_duration_l1818_181848


namespace geometric_arithmetic_progression_problem_l1818_181838

theorem geometric_arithmetic_progression_problem :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ * c₁ = b₁^2 ∧ a₁ + c₁ = 2*(b₁ + 8) ∧ a₁ * (c₁ + 64) = (b₁ + 8)^2) ∧
    (a₂ * c₂ = b₂^2 ∧ a₂ + c₂ = 2*(b₂ + 8) ∧ a₂ * (c₂ + 64) = (b₂ + 8)^2) ∧
    (a₁ = 4/9 ∧ b₁ = -20/9 ∧ c₁ = 100/9) ∧
    (a₂ = 4 ∧ b₂ = 12 ∧ c₂ = 36) :=
by sorry

end geometric_arithmetic_progression_problem_l1818_181838


namespace paper_width_problem_l1818_181897

theorem paper_width_problem (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 13)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 8.5 := by
  sorry

end paper_width_problem_l1818_181897


namespace range_of_a_l1818_181867

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = x^2) ∧
  (∀ x ≥ 0, deriv f x - x - 1 < 0)

/-- The main theorem -/
theorem range_of_a (f : ℝ → ℝ) (h : special_function f) :
  ∀ a, (f (2 - a) ≥ f a + 4 - 4*a) → a ≥ 1 :=
by sorry

end range_of_a_l1818_181867


namespace horner_method_v3_l1818_181831

-- Define the polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := ((((x + 0)*x + 2)*x + 3)*x + 1)*x + 1

-- Define v_3 as the result of Horner's method at x = 3
def v_3 : ℝ := horner 3

-- Theorem statement
theorem horner_method_v3 : v_3 = 36 := by sorry

end horner_method_v3_l1818_181831


namespace f_increasing_f_odd_range_l1818_181864

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Theorem 1: f(x) is an increasing function on ℝ
theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

-- Theorem 2: When f(x) is an odd function, its range on [-1, 2] is [-1/6, 3/10]
theorem f_odd_range (a : ℝ) 
  (h_odd : ∀ x : ℝ, f a (-x) = -(f a x)) : 
  Set.range (fun x => f a x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc (-1/6 : ℝ) (3/10) :=
sorry

end

end f_increasing_f_odd_range_l1818_181864


namespace sphere_section_distance_l1818_181851

theorem sphere_section_distance (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 2 →
  A = π →
  d = Real.sqrt 3 :=
by sorry

end sphere_section_distance_l1818_181851


namespace initial_weight_of_beef_l1818_181815

/-- The weight of a side of beef after five stages of processing --/
def final_weight (W : ℝ) : ℝ :=
  ((((W * 0.8) * 0.7) * 0.75) - 15) * 0.88

/-- Theorem stating the initial weight of the side of beef --/
theorem initial_weight_of_beef :
  ∃ W : ℝ, W > 0 ∧ final_weight W = 570 ∧ W = 1578 := by
  sorry

end initial_weight_of_beef_l1818_181815


namespace sum_of_seventh_terms_l1818_181849

/-- Given two arithmetic sequences a and b, prove that a₇ + b₇ = 8 -/
theorem sum_of_seventh_terms (a b : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ d : ℝ, a (n + 1) - a n = d) →  -- a is an arithmetic sequence
  (∀ n : ℕ, ∃ d : ℝ, b (n + 1) - b n = d) →  -- b is an arithmetic sequence
  a 2 + b 2 = 3 →                            -- given condition
  a 4 + b 4 = 5 →                            -- given condition
  a 7 + b 7 = 8 :=                           -- conclusion to prove
by sorry

end sum_of_seventh_terms_l1818_181849


namespace greatest_common_divisor_360_90_under_60_l1818_181852

theorem greatest_common_divisor_360_90_under_60 : 
  ∃ (n : ℕ), n = 30 ∧ 
  n ∣ 360 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 360 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end greatest_common_divisor_360_90_under_60_l1818_181852


namespace unfair_coin_flip_probability_l1818_181836

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_flip_probability (p : ℚ) (h : p = 1/3) :
  let n : ℕ := 8
  let k : ℕ := 3
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 1792/6561 := by
  sorry

end unfair_coin_flip_probability_l1818_181836


namespace sixteenth_root_of_unity_l1818_181813

theorem sixteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 15 ∧ 
  (Complex.tan (π / 8) + Complex.I) / (Complex.tan (π / 8) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 16)) :=
sorry

end sixteenth_root_of_unity_l1818_181813


namespace integral_proof_l1818_181847

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (1/16) * log (abs (x - 2)) + (15/16) * log (abs (x + 2)) + (33*x + 34) / (4*(x + 2)^2)

theorem integral_proof (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) :
  deriv f x = (x^3 - 6*x^2 + 13*x - 6) / ((x - 2)*(x + 2)^3) :=
by sorry

end integral_proof_l1818_181847


namespace trapezoid_area_theorem_l1818_181893

/-- A trapezoid with mutually perpendicular diagonals -/
structure Trapezoid :=
  (height : ℝ)
  (diagonal : ℝ)
  (diagonals_perpendicular : Bool)

/-- The area of a trapezoid with given properties -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem: The area of a trapezoid with mutually perpendicular diagonals, 
    height 4, and one diagonal of length 5 is equal to 50/3 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.height = 4)
  (h2 : t.diagonal = 5)
  (h3 : t.diagonals_perpendicular = true) : 
  trapezoid_area t = 50 / 3 :=
sorry

end trapezoid_area_theorem_l1818_181893


namespace intersection_of_A_and_B_l1818_181826

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 1| > 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l1818_181826


namespace shaded_area_is_nine_eighths_pi_l1818_181808

/-- Represents a right triangle with circles at its vertices and an additional circle --/
structure TriangleWithCircles where
  -- Side lengths of the right triangle
  ac : ℝ
  ab : ℝ
  bc : ℝ
  -- Radius of circles at triangle vertices
  y : ℝ
  -- Radius of circle P
  x : ℝ
  -- Conditions
  right_triangle : ac^2 + ab^2 = bc^2
  side_ac : y + 2*x + y = ac
  side_ab : y + 4*x + y = ab
  area_ratio : (2*x)^2 = 4*x^2

/-- The shaded area in the triangle configuration is 9π/8 square units --/
theorem shaded_area_is_nine_eighths_pi (t : TriangleWithCircles)
  (h1 : t.ac = 3)
  (h2 : t.ab = 4)
  (h3 : t.bc = 5) :
  3 * (π * t.y^2 / 2) + π * t.x^2 / 2 = 9 * π / 8 := by
  sorry

end shaded_area_is_nine_eighths_pi_l1818_181808


namespace jackson_collection_l1818_181805

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) (dollars_per_starfish : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  let sand_dollars := starfish * dollars_per_starfish
  hermit_crabs + spiral_shells + starfish + sand_dollars

/-- Theorem stating that Jackson's collection totals 3672 souvenirs -/
theorem jackson_collection :
  total_souvenirs 72 5 3 2 = 3672 := by
  sorry


end jackson_collection_l1818_181805


namespace sequence_formula_l1818_181868

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 2 * n.val - a n) : 
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) := by
sorry

end sequence_formula_l1818_181868


namespace trapezoid_in_isosceles_triangle_l1818_181894

/-- An isosceles triangle with a trapezoid inscribed within it. -/
structure IsoscelesTriangleWithTrapezoid where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- The distance from the apex to point D on side AB -/
  x : ℝ
  /-- The perimeter of the inscribed trapezoid -/
  trapezoidPerimeter : ℝ

/-- Theorem stating the condition for the inscribed trapezoid in an isosceles triangle -/
theorem trapezoid_in_isosceles_triangle 
    (t : IsoscelesTriangleWithTrapezoid) 
    (h1 : t.base = 12) 
    (h2 : t.side = 18) 
    (h3 : t.trapezoidPerimeter = 40) : 
    t.x = 6 := by
  sorry


end trapezoid_in_isosceles_triangle_l1818_181894


namespace ellipse_area_l1818_181845

/-- The area of an ellipse with semi-major axis a and semi-minor axis b is k*π where k = a*b -/
theorem ellipse_area (a b : ℝ) (h1 : a = 12) (h2 : b = 6) : ∃ k : ℝ, k = 72 ∧ a * b * π = k * π := by
  sorry

end ellipse_area_l1818_181845


namespace bryan_mineral_samples_l1818_181834

/-- The number of mineral samples Bryan has left after rearrangement -/
def samples_left (initial_samples_per_shelf : ℕ) (num_shelves : ℕ) (removed_per_shelf : ℕ) : ℕ :=
  (initial_samples_per_shelf - removed_per_shelf) * num_shelves

/-- Theorem stating the number of samples left after Bryan's rearrangement -/
theorem bryan_mineral_samples :
  samples_left 128 13 2 = 1638 := by
  sorry

end bryan_mineral_samples_l1818_181834


namespace find_k_l1818_181803

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ∀ (r : ℝ), e₁ ≠ r • e₂)

-- Define the vectors AB, CD, and CB
variable (k : ℝ)
def AB := 2 • e₁ + k • e₂
def CD := 2 • e₁ - 1 • e₂
def CB := 1 • e₁ + 3 • e₂

-- Define collinearity of A, B, and D
def collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

-- State the theorem
theorem find_k : 
  collinear (AB e₁ e₂ k) (CD e₁ e₂ - CB e₁ e₂) → k = -8 :=
sorry

end find_k_l1818_181803


namespace meeting_point_closer_to_a_l1818_181860

/-- The distance between two points A and B -/
def total_distance : ℕ := 85

/-- The constant speed of the person starting from point A -/
def speed_a : ℕ := 5

/-- The initial speed of the person starting from point B -/
def initial_speed_b : ℕ := 4

/-- The hourly increase in speed for the person starting from point B -/
def speed_increase_b : ℕ := 1

/-- The number of hours until the two people meet -/
def meeting_time : ℕ := 6

/-- The distance walked by the person starting from point A -/
def distance_a : ℕ := speed_a * meeting_time

/-- The distance walked by the person starting from point B -/
def distance_b : ℕ := meeting_time * (initial_speed_b + (meeting_time - 1) / 2 * speed_increase_b)

/-- The difference in distances walked by the two people -/
def distance_difference : ℤ := distance_b - distance_a

theorem meeting_point_closer_to_a : distance_difference = 9 := by sorry

end meeting_point_closer_to_a_l1818_181860


namespace number_of_factors_19368_l1818_181840

theorem number_of_factors_19368 : Nat.card (Nat.divisors 19368) = 24 := by
  sorry

end number_of_factors_19368_l1818_181840


namespace charity_run_donation_l1818_181880

theorem charity_run_donation (total_donation : ℕ) (race_length : ℕ) : 
  race_length = 5 ∧ 
  total_donation = 310 ∧ 
  (∃ initial_donation : ℕ, 
    total_donation = initial_donation * (2^race_length - 1)) →
  ∃ initial_donation : ℕ, initial_donation = 10 ∧
    total_donation = initial_donation * (2^race_length - 1) :=
by sorry

end charity_run_donation_l1818_181880


namespace symmetric_axis_of_quadratic_function_l1818_181827

/-- The symmetric axis of a quadratic function -/
def symmetric_axis (f : ℝ → ℝ) : ℝ := sorry

/-- A quadratic function in factored form -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a) * (x + b)

theorem symmetric_axis_of_quadratic_function :
  ∀ (f : ℝ → ℝ), f = quadratic_function 3 5 →
  symmetric_axis f = -1 := by sorry

end symmetric_axis_of_quadratic_function_l1818_181827


namespace number_puzzle_l1818_181858

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 4 = x / 4 - 10) ∧ x = 280 := by
  sorry

end number_puzzle_l1818_181858


namespace people_to_left_of_kolya_l1818_181825

/-- Represents a person in the line -/
structure Person where
  name : String

/-- Represents the arrangement of people in a line -/
structure Arrangement where
  people : List Person
  kolya_index : Nat
  sasha_index : Nat

/-- The number of people to the right of a person at a given index -/
def peopleToRight (arr : Arrangement) (index : Nat) : Nat :=
  arr.people.length - index - 1

/-- The number of people to the left of a person at a given index -/
def peopleToLeft (arr : Arrangement) (index : Nat) : Nat :=
  index

theorem people_to_left_of_kolya (arr : Arrangement) 
  (h1 : peopleToRight arr arr.kolya_index = 12)
  (h2 : peopleToLeft arr arr.sasha_index = 20)
  (h3 : peopleToRight arr arr.sasha_index = 8) :
  peopleToLeft arr arr.kolya_index = 16 := by
  sorry

end people_to_left_of_kolya_l1818_181825


namespace max_value_sum_fractions_l1818_181832

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_sum : a + b + c = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 / 2 :=
sorry

end max_value_sum_fractions_l1818_181832


namespace problem_1_problem_2_l1818_181807

-- Problem 1
theorem problem_1 : 
  Real.sqrt 12 + |(-4)| - (2003 - Real.pi)^0 - 2 * Real.cos (30 * π / 180) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℤ) (h1 : 0 < a) (h2 : a < 4) (h3 : a ≠ 2) : 
  (a + 2 - 5 / (a - 2)) / ((3 - a) / (2 * a - 4)) = -2 * a - 6 := by
  sorry

end problem_1_problem_2_l1818_181807


namespace remainder_3572_div_49_l1818_181824

theorem remainder_3572_div_49 : 3572 % 49 = 44 := by
  sorry

end remainder_3572_div_49_l1818_181824


namespace solution_of_equation_l1818_181886

theorem solution_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end solution_of_equation_l1818_181886


namespace batsman_average_increase_l1818_181835

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  notOutCount : ℕ

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : ℚ :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (before after : Batsman) : ℚ :=
  battingAverage after - battingAverage before

theorem batsman_average_increase :
  ∀ (before : Batsman),
    before.innings = 19 →
    before.notOutCount = 0 →
    let after : Batsman :=
      { innings := 20
      , totalScore := before.totalScore + 90
      , notOutCount := 0
      }
    battingAverage after = 52 →
    averageIncrease before after = 2 := by
  sorry

end batsman_average_increase_l1818_181835


namespace minimum_cats_with_stripes_and_black_ear_l1818_181816

theorem minimum_cats_with_stripes_and_black_ear (total_cats : ℕ) (mice_catchers : ℕ) 
  (striped_cats : ℕ) (black_ear_cats : ℕ) 
  (h1 : total_cats = 66) (h2 : mice_catchers = 21) 
  (h3 : striped_cats = 32) (h4 : black_ear_cats = 27) : 
  ∃ (x : ℕ), x = 14 ∧ 
  x ≤ striped_cats ∧ 
  x ≤ black_ear_cats ∧
  x ≤ total_cats - mice_catchers ∧
  ∀ (y : ℕ), y < x → 
    y > striped_cats + black_ear_cats - (total_cats - mice_catchers) := by
  sorry

end minimum_cats_with_stripes_and_black_ear_l1818_181816


namespace tom_jogging_distance_l1818_181874

/-- The distance Tom jogs in 15 minutes given his rate -/
theorem tom_jogging_distance (rate : ℝ) (time : ℝ) (h1 : rate = 1 / 18) (h2 : time = 15) :
  rate * time = 5 / 6 := by
  sorry

end tom_jogging_distance_l1818_181874


namespace binomial_expansion_terms_l1818_181857

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 : ℝ) * x^(n-1) * a = 56 ∧
  (Nat.choose n 2 : ℝ) * x^(n-2) * a^2 = 168 ∧
  (Nat.choose n 3 : ℝ) * x^(n-3) * a^3 = 336 →
  n = 3 := by
sorry

end binomial_expansion_terms_l1818_181857


namespace area_triangle_ABC_area_DEFGH_area_triangle_JKL_l1818_181866

-- Define the grid unit
def grid_unit : ℝ := 1

-- Define the dimensions of triangle ABC
def triangle_ABC_base : ℝ := 2 * grid_unit
def triangle_ABC_height : ℝ := 3 * grid_unit

-- Define the dimensions of the square for DEFGH and JKL
def square_side : ℝ := 5 * grid_unit

-- Theorem for the area of triangle ABC
theorem area_triangle_ABC : 
  (1/2) * triangle_ABC_base * triangle_ABC_height = 3 := by sorry

-- Theorem for the area of figure DEFGH
theorem area_DEFGH : 
  square_side^2 - (1/2) * triangle_ABC_base * triangle_ABC_height = 22 := by sorry

-- Theorem for the area of triangle JKL
theorem area_triangle_JKL : 
  square_side^2 - ((1/2) * triangle_ABC_base * triangle_ABC_height + 
  (1/2) * square_side * (square_side - triangle_ABC_height) + 
  (1/2) * square_side * triangle_ABC_base) = 19/2 := by sorry

end area_triangle_ABC_area_DEFGH_area_triangle_JKL_l1818_181866


namespace new_crew_member_weight_l1818_181895

/-- Given a crew of oarsmen, prove that replacing a crew member results in a specific weight for the new crew member. -/
theorem new_crew_member_weight
  (n : ℕ) -- Number of oarsmen
  (avg_increase : ℝ) -- Increase in average weight
  (old_weight : ℝ) -- Weight of the replaced crew member
  (h1 : n = 20) -- There are 20 oarsmen
  (h2 : avg_increase = 2) -- Average weight increases by 2 kg
  (h3 : old_weight = 40) -- The replaced crew member weighs 40 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by sorry

end new_crew_member_weight_l1818_181895


namespace simplify_expression_l1818_181844

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end simplify_expression_l1818_181844


namespace circle_path_in_right_triangle_l1818_181892

theorem circle_path_in_right_triangle (a b c : ℝ) (r : ℝ) :
  a = 5 →
  b = 12 →
  c = 13 →
  r = 2 →
  a^2 + b^2 = c^2 →
  let path_length := (a - 2*r) + (b - 2*r) + (c - 2*r)
  path_length = 9 := by
  sorry

end circle_path_in_right_triangle_l1818_181892


namespace new_job_wage_is_15_l1818_181829

/-- Represents the wage scenario for Maisy's job options -/
structure WageScenario where
  current_hours : ℕ
  current_wage : ℕ
  new_hours : ℕ
  new_bonus : ℕ
  earnings_difference : ℕ

/-- Calculates the wage per hour for the new job -/
def new_job_wage (scenario : WageScenario) : ℕ :=
  (scenario.current_hours * scenario.current_wage + scenario.earnings_difference - scenario.new_bonus) / scenario.new_hours

/-- Theorem stating that given the specified conditions, the new job wage is $15 per hour -/
theorem new_job_wage_is_15 (scenario : WageScenario) 
  (h1 : scenario.current_hours = 8)
  (h2 : scenario.current_wage = 10)
  (h3 : scenario.new_hours = 4)
  (h4 : scenario.new_bonus = 35)
  (h5 : scenario.earnings_difference = 15) :
  new_job_wage scenario = 15 := by
  sorry

#eval new_job_wage { current_hours := 8, current_wage := 10, new_hours := 4, new_bonus := 35, earnings_difference := 15 }

end new_job_wage_is_15_l1818_181829


namespace x_plus_y_value_l1818_181843

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 := by
  sorry

end x_plus_y_value_l1818_181843


namespace sum_of_digits_next_l1818_181859

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 876) : S (n + 1) = 877 := by
  sorry

end sum_of_digits_next_l1818_181859


namespace mrs_hilt_apple_consumption_l1818_181889

-- Define the rate of apple consumption
def apples_per_hour : ℕ := 10

-- Define the number of hours
def total_hours : ℕ := 6

-- Theorem to prove
theorem mrs_hilt_apple_consumption :
  apples_per_hour * total_hours = 60 :=
by sorry

end mrs_hilt_apple_consumption_l1818_181889


namespace problem_1_problem_2_l1818_181899

-- Problem 1
theorem problem_1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a ≠ 1) : (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) := by
  sorry

end problem_1_problem_2_l1818_181899


namespace prob_third_given_a_wins_l1818_181850

/-- The probability of Player A winning a single game -/
def p : ℚ := 2/3

/-- The probability of Player A winning the championship -/
def prob_a_wins : ℚ := p^2 + 2*p^2*(1-p)

/-- The probability of the match going to the third game and Player A winning -/
def prob_third_and_a_wins : ℚ := 2*p^2*(1-p)

/-- The probability of the match going to the third game given that Player A wins the championship -/
theorem prob_third_given_a_wins : 
  prob_third_and_a_wins / prob_a_wins = 2/5 := by sorry

end prob_third_given_a_wins_l1818_181850
