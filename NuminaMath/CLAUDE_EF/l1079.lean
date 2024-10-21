import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l1079_107916

/-- Definition of a parabola (simplified for this context) -/
def IsParabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0

/-- The equation x^2 + ky^2 = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : ¬ IsParabola (fun x y => x^2 + k*y^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l1079_107916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_shopping_payments_l1079_107952

/-- Calculates the discounted price based on the given rules -/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 100 then price
  else if price ≤ 300 then price * 0.9
  else 300 * 0.9 + (price - 300) * 0.8

/-- Represents Mr. Wang's shopping scenario -/
structure ShoppingScenario where
  firstTripPrice : ℝ
  secondTripPrice : ℝ
  combinedSavings : ℝ
  totalWithoutDiscount : ℝ

/-- Theorem stating the correct payments for Mr. Wang's shopping trips -/
theorem wang_shopping_payments (scenario : ShoppingScenario) :
  scenario.firstTripPrice = 190 ∧
  scenario.secondTripPrice = 390 ∧
  discountedPrice scenario.firstTripPrice = 171 ∧
  discountedPrice scenario.secondTripPrice = 342 ∧
  (discountedPrice scenario.firstTripPrice + discountedPrice scenario.secondTripPrice) -
    discountedPrice (scenario.firstTripPrice + scenario.secondTripPrice) = 19 ∧
  (scenario.firstTripPrice + scenario.secondTripPrice) -
    (discountedPrice scenario.firstTripPrice + discountedPrice scenario.secondTripPrice) = 67 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_shopping_payments_l1079_107952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_M_l1079_107965

/-- The number of positive divisors of M = 2^5 · 3^4 · 5^2 · 7^3 · 11^1 -/
def M : ℕ := 2^5 * 3^4 * 5^2 * 7^3 * 11^1

/-- Theorem: The number of positive divisors of M is 720 -/
theorem number_of_divisors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_M_l1079_107965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_radius_l1079_107932

/-- 
Given a triangle with sides a, b, c, and semiperimeter p,
prove that the radius of the inscribed circle is maximized
when the triangle is equilateral, and the maximum radius is p / √27.
-/
theorem max_inscribed_circle_radius 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (r : ℝ) (hr : r = Real.sqrt (p * (p - a) * (p - b) * (p - c)) / p) :
  r ≤ p / Real.sqrt 27 ∧ 
  (r = p / Real.sqrt 27 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_radius_l1079_107932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1079_107984

noncomputable def f (x : Real) : Real := 
  Real.sqrt 3 * Real.cos (2 * x) + 2 * (Real.cos (Real.pi / 4 - x))^2 - 1

theorem f_properties :
  (∃ (T : Real), T > 0 ∧ 
    (∀ (x : Real), f (x + T) = f x) ∧ 
    (∀ (S : Real), S > 0 ∧ (∀ (x : Real), f (x + S) = f x) → T ≤ S)) ∧
  (∀ (y : Real), (∃ (x : Real), x ∈ Set.Icc (-Real.pi/3) (Real.pi/2) ∧ f x = y) ↔ 
    y ∈ Set.Icc (-Real.sqrt 3) 2) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1079_107984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_car_time_l1079_107997

/-- The time (in seconds) for a train to completely pass a car traveling in the same direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (car_speed : ℝ) : ℝ :=
  train_length / ((train_speed - car_speed) * (5/18))

theorem train_passing_car_time :
  let train_length : ℝ := 150
  let train_speed : ℝ := 75
  let car_speed : ℝ := 45
  ⌊train_passing_time train_length train_speed car_speed⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_car_time_l1079_107997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_minimum_values_l1079_107926

noncomputable section

-- Define the line passing through P(1, 3) and intersecting positive semi-axes
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 3 = k * (p.1 - 1)}

-- Define points A and B as intersections with axes
def A (k : ℝ) : ℝ × ℝ := (0, 3 - k)
def B (k : ℝ) : ℝ × ℝ := (1 - 3/k, 0)

-- Define the area of triangle AOB
def triangleArea (k : ℝ) : ℝ := 1/2 * (3 - k) * (1 - 3/k)

-- Define OA + OB
def sumOAOB (k : ℝ) : ℝ := (3 - k) + (1 - 3/k)

-- Define PA•PB
def productPAPB (α : ℝ) : ℝ := -3 / (Real.sin α * Real.cos α)

theorem line_equation_and_minimum_values 
  (k : ℝ) (h1 : k < 0) (h2 : triangleArea k = 6) :
  (∃ l : Set (ℝ × ℝ), l = Line (-3)) ∧ 
  (∃ min : ℝ, min = 4 + 2 * Real.sqrt 3 ∧ ∀ k', sumOAOB k' ≥ min) ∧
  (∃ α : ℝ, productPAPB α = - 3 * Real.sqrt 2 ∧ 
    ∀ β, π/2 < β ∧ β < π → productPAPB β ≥ productPAPB α) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_minimum_values_l1079_107926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1079_107975

theorem train_length_calculation (train_speed_kmph : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed_kmph = 54 →
  time_to_cross = 53.66237367677253 →
  bridge_length = 660 →
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * time_to_cross
  let train_length := total_distance - bridge_length
  train_length = 144.93560565158795 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1079_107975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ATU_implies_t_eq_5_l1079_107980

noncomputable section

/-- Triangle ABC with vertices A(1, 10), B(3, 0), C(10, 0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨1, 10⟩, ⟨3, 0⟩, ⟨10, 0⟩}

/-- Point T on line segment AB -/
def point_T (t : ℝ) : ℝ × ℝ :=
  ⟨3 - t/5, t⟩

/-- Point U on line segment AC -/
def point_U (t : ℝ) : ℝ × ℝ :=
  ⟨(190 - 9*t)/10, t⟩

/-- Area of triangle ATU -/
def area_ATU (t : ℝ) : ℝ :=
  (1/2) * (abs ((190 - 9*t)/10 - (3 - t/5))) * (10 - t)

/-- Theorem: If the area of triangle ATU is 18, then t = 5 -/
theorem area_ATU_implies_t_eq_5 :
  ∀ t : ℝ, area_ATU t = 18 → t = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ATU_implies_t_eq_5_l1079_107980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1079_107905

noncomputable def vector_a : ℝ × ℝ := (3, -4)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_c (y : ℝ) : ℝ × ℝ := (2, y)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_problem (x y : ℝ) :
  parallel vector_a (vector_b x) →
  perpendicular vector_a (vector_c y) →
  vector_b x = (2, -8/3) ∧
  vector_c y = (2, 3/2) ∧
  angle (vector_b x) (vector_c y) = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1079_107905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2012_l1079_107931

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.log x / Real.log 2) + b * (Real.log x / Real.log 3) + 2

-- State the theorem
theorem f_value_at_2012 (a b : ℝ) :
  f a b (1/2012) = 5 → f a b 2012 = -1 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2012_l1079_107931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l1079_107943

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
noncomputable def decaf_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
                     (new_stock : ℝ) (new_decaf_percent : ℝ) : ℝ :=
  let initial_decaf := initial_stock * initial_decaf_percent / 100
  let new_decaf := new_stock * new_decaf_percent / 100
  let total_decaf := initial_decaf + new_decaf
  let total_stock := initial_stock + new_stock
  (total_decaf / total_stock) * 100

/-- Theorem: The percentage of decaffeinated coffee in the grocer's total stock is 44% -/
theorem grocer_decaf_percentage :
  decaf_percentage 400 40 100 60 = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l1079_107943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1079_107938

/-- An arithmetic progression with first three terms 2x - 3, 3x - 1, 4x + 1 -/
def arithmetic_progression (x : ℤ) : ℕ → ℤ := λ n => (n + 1) * x - (5 - n)

theorem unique_solution :
  ∃! x : ℤ, 
    (∀ n : ℕ, n < 3 → arithmetic_progression x (n + 1) - arithmetic_progression x n = arithmetic_progression x 1 - arithmetic_progression x 0) ∧ 
    (∀ n : ℕ, n < 3 → arithmetic_progression x n > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1079_107938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_nine_cards_reverse_fiftytwo_cards_l1079_107988

/-- Represents a deck of cards as a permutation of integers -/
def Deck (n : ℕ) := Fin n → Fin n

/-- Checks if a deck is in reverse order -/
def is_reversed {n : ℕ} (d : Deck n) : Prop :=
  ∀ i j : Fin n, i ≤ j → d i ≥ d j

/-- Represents a single operation on the deck -/
inductive Operation (n : ℕ)
| move : Fin n → Fin n → Fin n → Operation n

/-- Applies an operation to a deck -/
def apply_operation {n : ℕ} (d : Deck n) (op : Operation n) : Deck n :=
  sorry

/-- Checks if a sequence of operations reverses the deck -/
def reverses_deck {n : ℕ} (d : Deck n) (ops : List (Operation n)) : Prop :=
  is_reversed (ops.foldl apply_operation d)

/-- The minimum number of operations required to reverse a deck -/
noncomputable def min_operations (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For a deck of 9 cards, it can be reversed in at most 5 operations -/
theorem reverse_nine_cards :
  min_operations 9 ≤ 5 := by
  sorry

/-- Theorem: For a deck of 52 cards, it takes exactly 27 operations to reverse -/
theorem reverse_fiftytwo_cards :
  min_operations 52 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_nine_cards_reverse_fiftytwo_cards_l1079_107988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1079_107924

/-- Represents the time taken to complete a work -/
def WorkTime := ℝ

/-- Represents the rate of work completion -/
def WorkRate := ℝ

/-- The total amount of work to be done -/
def TotalWork := ℝ

theorem work_completion_time 
  (total_work : TotalWork)
  (time_together : WorkTime)
  (time_a_alone : WorkTime)
  (h1 : time_together = (4 : ℝ))
  (h2 : time_a_alone = (8 : ℝ))
  : time_together = (4 : ℝ) :=
by
  -- The proof goes here
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1079_107924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_x_plus_one_l1079_107971

open Real
open Set
open MeasureTheory

/-- The Fourier series expansion of f(x) = x + 1 on (0, π) -/
theorem fourier_series_x_plus_one (x : ℝ) (hx : x ∈ Ioo 0 π) :
  let f : ℝ → ℝ := λ x => x + 1
  let a : ℕ → ℝ := λ n => 4 * (π + 1) / π * ((-1)^n / (2 * ↑n + 1))
  let series := λ x => ∑' n, a n * Real.cos ((2 * ↑n + 1) * x / 2)
  f x = series x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_x_plus_one_l1079_107971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_2_l1079_107995

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 / (x - 2)) - (4 / (x^2 - 4))

-- State the theorem
theorem limit_of_f_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 1/4| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_2_l1079_107995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_number_proof_l1079_107912

/-- The smallest positive integer not divisible by 2 or 3 and not expressible as |2^a - 3^b| -/
def smallest_special_number : ℕ := 35

theorem smallest_special_number_proof :
  (∀ k : ℕ, 0 < k → k < smallest_special_number →
    (k % 2 = 0 ∨ k % 3 = 0 ∨ ∃ a b : ℕ, Int.natAbs (2^a - 3^b) = k)) ∧
  smallest_special_number % 2 ≠ 0 ∧
  smallest_special_number % 3 ≠ 0 ∧
  ∀ a b : ℕ, Int.natAbs (2^a - 3^b) ≠ smallest_special_number :=
by sorry

#check smallest_special_number_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_number_proof_l1079_107912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_negative_in_second_quadrant_l1079_107929

theorem sin_cos_negative_in_second_quadrant (θ : Real) :
  π / 2 < θ ∧ θ < π → Real.sin (Real.cos θ) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_negative_in_second_quadrant_l1079_107929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_3_50_l1079_107964

/-- Represents the outcome of rolling a fair 8-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- The probability of each outcome for a fair 8-sided die -/
noncomputable def prob (roll : DieRoll) : ℝ := 1 / 8

/-- The winnings for a given roll -/
noncomputable def winnings (roll : DieRoll) : ℝ :=
  match roll with
  | DieRoll.one => 1 / 2
  | DieRoll.two => 2
  | DieRoll.three => 3 / 2
  | DieRoll.four => 4
  | DieRoll.five => 5 / 2
  | DieRoll.six => 6
  | DieRoll.seven => 7 / 2
  | DieRoll.eight => 8

/-- The expected value of winnings -/
noncomputable def expected_value : ℝ :=
  (prob DieRoll.one * winnings DieRoll.one) +
  (prob DieRoll.two * winnings DieRoll.two) +
  (prob DieRoll.three * winnings DieRoll.three) +
  (prob DieRoll.four * winnings DieRoll.four) +
  (prob DieRoll.five * winnings DieRoll.five) +
  (prob DieRoll.six * winnings DieRoll.six) +
  (prob DieRoll.seven * winnings DieRoll.seven) +
  (prob DieRoll.eight * winnings DieRoll.eight)

theorem expected_value_is_3_50 : expected_value = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_3_50_l1079_107964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1079_107970

/-- Calculate the simple interest rate given principal, interest, and time -/
noncomputable def calculate_interest_rate (principal : ℝ) (interest : ℝ) (time : ℝ) : ℝ :=
  (interest / (principal * time)) * 100

/-- Theorem: Given the specified conditions, the interest rate is 3.5% -/
theorem interest_rate_calculation :
  let principal : ℝ := 400
  let interest : ℝ := 70
  let time : ℝ := 5
  calculate_interest_rate principal interest time = 3.5 := by
  sorry

/-- Evaluate the interest rate calculation -/
def evaluate_interest_rate : ℚ :=
  (70 / (400 * 5)) * 100

#eval evaluate_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1079_107970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_comparison_l1079_107901

/-- Represents a school with its test scores -/
structure School where
  scores : List ℝ

/-- Calculates the average of a list of real numbers -/
noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

/-- Calculates the variance of a list of real numbers -/
noncomputable def variance (l : List ℝ) : ℝ :=
  let avg := average l
  (l.map (λ x => (x - avg) ^ 2)).sum / l.length

/-- States that one list of scores is more uniform than another -/
def more_uniform (s1 s2 : List ℝ) : Prop :=
  ∀ x ∈ s1, ∀ y ∈ s2, |x - average s1| ≤ |y - average s2|

/-- Theorem stating that if School A's scores are more uniform than School B's,
    and they have the same average, then School A's variance is less than School B's -/
theorem variance_comparison (A B : School) 
    (h1 : average A.scores = average B.scores)
    (h2 : more_uniform A.scores B.scores) :
    variance A.scores < variance B.scores := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_comparison_l1079_107901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_150_meters_l1079_107967

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to cross the man. -/
noncomputable def train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_speed + man_speed) * crossing_time * (1000 / 3600)

/-- Theorem stating that the length of the train is 150 meters under the given conditions. -/
theorem train_length_is_150_meters :
  train_length 85 5 6 = 150 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  simp [mul_add, mul_assoc]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_150_meters_l1079_107967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt3_and_3pi_l1079_107972

theorem whole_numbers_between_sqrt3_and_3pi :
  (Finset.range (Int.toNat (Int.floor (3 * Real.pi) - Int.ceil (Real.sqrt 3) + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt3_and_3pi_l1079_107972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_huanhuan_lucheng_third_l1079_107903

-- Define the contestants
inductive Contestant : Type
| Beibei : Contestant
| Jingjing : Contestant
| Huanhuan : Contestant

-- Define the cities
inductive City : Type
| Lucheng : City
| Yongjia : City
| Ruian : City

-- Define the prizes
inductive Prize : Type
| First : Prize
| Second : Prize
| Third : Prize

-- Define the function that assigns a city to each contestant
variable (hometown : Contestant → City)

-- Define the function that assigns a prize to each contestant
variable (prize_won : Contestant → Prize)

-- State the theorem
theorem huanhuan_lucheng_third : 
  (hometown Contestant.Beibei ≠ City.Lucheng) →
  (hometown Contestant.Jingjing ≠ City.Yongjia) →
  (∀ c : Contestant, hometown c = City.Lucheng → prize_won c ≠ Prize.First) →
  (∃ c : Contestant, hometown c = City.Yongjia ∧ prize_won c = Prize.Second) →
  (prize_won Contestant.Jingjing ≠ Prize.Third) →
  (hometown Contestant.Huanhuan = City.Lucheng ∧ prize_won Contestant.Huanhuan = Prize.Third) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_huanhuan_lucheng_third_l1079_107903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_MN_l1079_107969

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 1]
noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 1/2]

theorem inverse_of_MN :
  (M * N)⁻¹ = !![1/3, 0; 0, 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_MN_l1079_107969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_distance_to_origin_l1079_107950

-- Define the complex sequence
noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => (z n)^2 + Complex.I

-- State the theorem
theorem z_111_distance_to_origin : Complex.abs (z 110) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_distance_to_origin_l1079_107950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_35_l1079_107919

def T_shape : Type := List ℕ × List ℕ

def is_valid_T_shape (t : T_shape) : Prop :=
  t.1.length = 4 ∧ t.2.length = 5 ∧ t.1.get? 1 = t.2.get? 1

def digits_in_range (t : T_shape) : Prop :=
  ∀ n ∈ t.1 ++ t.2, 1 ≤ n ∧ n ≤ 9

def all_different (t : T_shape) : Prop :=
  (t.1 ++ t.2).Nodup

def vertical_sum (t : T_shape) : ℕ :=
  t.1.sum

def horizontal_sum (t : T_shape) : ℕ :=
  t.2.sum

theorem sum_of_digits_is_35 (t : T_shape) :
  is_valid_T_shape t →
  digits_in_range t →
  all_different t →
  vertical_sum t = 26 →
  horizontal_sum t = 20 →
  (t.1 ++ t.2).sum = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_35_l1079_107919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_comparison_l1079_107961

theorem equality_comparison :
  (-(0.3) = -0.3) ∧ 
  (3^2 ≠ 2^3) ∧ 
  (-(-3) ≠ -(abs (-3))) ∧ 
  ((-2)^2 ≠ -(2^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_comparison_l1079_107961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_with_buckets_l1079_107990

noncomputable section

/-- The volume of a sphere with radius r -/
def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The number of spherical buckets needed to fill a cylindrical tank -/
def bucketsNeeded (bucketRadius tankRadius tankHeight : ℝ) : ℝ :=
  cylinderVolume tankRadius tankHeight / sphereVolume bucketRadius

theorem fill_tank_with_buckets :
  bucketsNeeded 8 8 32 = 3 := by
  -- Unfold the definitions
  unfold bucketsNeeded cylinderVolume sphereVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_with_buckets_l1079_107990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_zero_l1079_107923

/-- Line 1 is defined by the point (1, 2, -1) and direction vector (-1, 3, 2) -/
def line1 (u : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1 - u
  | 1 => 2 + 3*u
  | 2 => -1 + 2*u

/-- Line 2 is defined by the point (3, -1, 5) and direction vector (2, 1, -3) -/
def line2 (v : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 2*v
  | 1 => -1 + v
  | 2 => 5 - 3*v

/-- The squared distance between two points -/
def distance_squared (p q : Fin 3 → ℝ) : ℝ :=
  (p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2

/-- The shortest distance between the two lines is 0 -/
theorem shortest_distance_zero :
  ∃ u v : ℝ, distance_squared (line1 u) (line2 v) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_zero_l1079_107923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_existence_l1079_107940

theorem modular_inverse_existence (p : ℕ) (a : ℕ) (h_prime : Nat.Prime p) (h_not_div : ¬(p ∣ a)) :
  ∃ b : ℕ, (a * b) % p = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_existence_l1079_107940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_32458_l1079_107977

def n : ℕ := 32458

theorem divisors_of_32458 :
  (Finset.filter (λ i => n % i = 0) (Finset.range 10 \ {0})).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_32458_l1079_107977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l1079_107989

/-- A line y = x + a that does not pass through the second quadrant -/
structure Line where
  a : ℝ
  not_in_second_quadrant : ∀ x y : ℝ, y = x + a → (x < 0 → y ≤ 0)

/-- The number of real solutions to a quadratic equation -/
noncomputable def num_real_solutions (a b c : ℝ) : ℕ :=
  if a = 0 then
    if b ≠ 0 then 1 else 0
  else
    let discriminant := b^2 - 4*a*c
    if discriminant > 0 then 2
    else if discriminant = 0 then 1
    else 0

/-- Theorem stating the number of real solutions for the given problem -/
theorem solutions_count (l : Line) :
  num_real_solutions l.a 2 1 = 1 ∨ num_real_solutions l.a 2 1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l1079_107989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1079_107973

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_ge_b : a ≥ b)

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The equation of a line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

theorem ellipse_and_line_intersection
  (e : Ellipse)
  (h_minor_axis : e.b = Real.sqrt 3)
  (h_eccentricity : e.eccentricity = 1/2)
  (l : Line)
  (h_line_passes : l.equation 0 3)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    e.equation x₁ y₁ ∧ e.equation x₂ y₂ ∧
    l.equation x₁ y₁ ∧ l.equation x₂ y₂ ∧
    x₁ = x₂/2 ∧ y₁ = (3 + y₂)/2) :
  e.equation = fun x y ↦ x^2/4 + y^2/3 = 1 ∧
  (l.slope = 3/2 ∨ l.slope = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1079_107973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_weight_calculations_l1079_107922

/-- The weight of a 5-jiao coin in grams -/
noncomputable def coin_weight : ℝ := 3

/-- Convert grams to kilograms -/
noncomputable def grams_to_kg (g : ℝ) : ℝ := g / 1000

/-- Convert grams to tons -/
noncomputable def grams_to_tons (g : ℝ) : ℝ := g / 1000000

theorem coin_weight_calculations :
  let weight_10k := grams_to_kg (10000 * coin_weight)
  let weight_10m := grams_to_tons (10000000 * coin_weight)
  let weight_200m := grams_to_tons (200000000 * coin_weight)
  weight_10k = 30 ∧ weight_10m = 30 ∧ weight_200m = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_weight_calculations_l1079_107922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_63_l1079_107937

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that 2 chairs and 1 table cost 60% of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that 1 chair and 1 table cost $72 -/
axiom total_price : chair_price + table_price = 72

theorem table_price_is_63 : table_price = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_63_l1079_107937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_term_is_1923_l1079_107960

def v : ℕ → ℕ
| 0 => 2
| n + 1 =>
  let k := ((n + 2) * (n + 3) / 2 - n - 1) / 3
  let r := n + 1 - (k * (k + 1) / 2)
  3 * k + r + 1

theorem thousandth_term_is_1923 : v 999 = 1923 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_term_is_1923_l1079_107960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1079_107959

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.3 0.4
  let b : ℝ := Real.log 0.3 / Real.log 4
  let c : ℝ := Real.rpow 4 0.3
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1079_107959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_4950_l1079_107925

/-- The fraction 1/9801 as a rational number -/
def fraction : ℚ := 1 / 9801

/-- The length of the period in the repeating decimal representation of 1/9801 -/
def period_length : ℕ := 200

/-- A function that returns the nth digit in the repeating decimal representation of 1/9801 -/
def digit (n : ℕ) : ℕ := sorry

/-- The sum of digits in one period of the repeating decimal representation of 1/9801 -/
def digit_sum : ℕ := Finset.sum (Finset.range period_length) digit

/-- Theorem stating that the sum of digits in one period is 4950 -/
theorem sum_of_digits_is_4950 : digit_sum = 4950 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_is_4950_l1079_107925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_quadrilateral_l1079_107946

/-- A convex quadrilateral with side lengths no more than 7 -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)
  side_lengths : ∀ i : Fin 4, dist (vertices i) (vertices (i.succ)) ≤ 7

/-- A circle with radius 5 -/
def Circle (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | dist p center ≤ 5}

/-- The theorem stating that four circles cover the quadrilateral -/
theorem circles_cover_quadrilateral (q : ConvexQuadrilateral) :
  Set.range q.vertices ⊆ ⋃ i : Fin 4, Circle (q.vertices i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_quadrilateral_l1079_107946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_on_line_and_ellipse_l1079_107981

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 2 / 2) * t - Real.sqrt 2, (Real.sqrt 2 / 4) * t)

-- Define the ellipse C
noncomputable def ellipse_C (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, (1 / 2) * Real.sin θ)

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ :=
  (-Real.sqrt 2 / 2, Real.sqrt 2 / 4)

-- Theorem statement
theorem intersection_point_is_on_line_and_ellipse :
  ∃ (t θ : ℝ), line_l t = intersection_point ∧ ellipse_C θ = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_on_line_and_ellipse_l1079_107981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l1079_107918

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, 2 * x^2 + b * x + 16 = 0 → x.re ≠ x.im) → 
  b > -8 * Real.sqrt 2 ∧ b < 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l1079_107918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_cosine_values_l1079_107947

theorem tangent_and_cosine_values (α : Real) 
  (h1 : Real.tan (π/4 + α) = 1/7)
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.tan α = -3/4 ∧ Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_cosine_values_l1079_107947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_l1079_107979

/-- The selling price of the article -/
noncomputable def selling_price : ℝ := 8.00

/-- The profit percentage on the selling price -/
noncomputable def profit_percentage : ℝ := 0.12

/-- The expenses percentage on the selling price -/
noncomputable def expenses_percentage : ℝ := 0.18

/-- Calculate the cost price given the selling price and percentages -/
noncomputable def cost_price : ℝ := selling_price * (1 - profit_percentage - expenses_percentage)

/-- Calculate the markup rate -/
noncomputable def markup_rate : ℝ := (selling_price - cost_price) / cost_price

/-- Theorem stating that the markup rate is approximately 42.857% -/
theorem markup_rate_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |markup_rate * 100 - 42.857| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_l1079_107979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_condition_l1079_107991

theorem tan_equality_condition (α β : ℝ) : 
  (¬∀ α β : ℝ, α = β → Real.tan α = Real.tan β) ∧ 
  (¬∀ α β : ℝ, Real.tan α = Real.tan β → α = β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_condition_l1079_107991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hershel_betta_fish_count_l1079_107956

theorem hershel_betta_fish_count :
  ∀ (initial_betta : ℕ),
  (let initial_goldfish : ℕ := 15;
   let bexley_betta : ℕ := (2 * initial_betta) / 5;
   let bexley_goldfish : ℕ := initial_goldfish / 3;
   let total_fish : ℕ := initial_betta + initial_goldfish + bexley_betta + bexley_goldfish;
   let remaining_fish : ℕ := total_fish / 2;
   remaining_fish = 17) → initial_betta = 10 :=
by
  intro initial_betta
  intro h
  sorry

#check hershel_betta_fish_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hershel_betta_fish_count_l1079_107956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1079_107945

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1079_107945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauliflower_area_is_one_square_foot_l1079_107999

/-- Represents a square garden where cauliflowers are grown -/
structure CauliflowerGarden where
  side : ℕ
  area_per_cauliflower : ℕ

/-- Calculates the number of cauliflowers in the garden -/
def cauliflowers_count (g : CauliflowerGarden) : ℕ :=
  g.side^2 / g.area_per_cauliflower

theorem cauliflower_area_is_one_square_foot 
  (g : CauliflowerGarden)
  (h1 : cauliflowers_count g = 40401)
  (h2 : cauliflowers_count g = cauliflowers_count { g with side := g.side } + 401) :
  g.area_per_cauliflower = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauliflower_area_is_one_square_foot_l1079_107999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_range_l1079_107974

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 / 3 - a / 2 * x^2 + x

theorem monotonic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x ≤ y → f a x ≥ f a y) →
  a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_range_l1079_107974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_B_l1079_107955

/-- Represents the natural number A as described in the problem -/
def A : ℕ := 12345678910111213 -- (continuing up to 99100)

/-- Represents the number of digits to be removed -/
def digitsToRemove : ℕ := 100

/-- Represents the sum of all digits in A -/
def sumOfDigitsA : ℕ := 901

/-- Represents the sum of all removed digits -/
def sumOfRemovedDigits : ℕ := 415

/-- Function to remove digits from a number -/
def removeDigits (n : ℕ) (digits : Finset ℕ) : ℕ := sorry

/-- Predicate to check if B is the smallest possible number after removing digits -/
def isSmallestPossible (B : ℕ) (A : ℕ) (digits : Finset ℕ) : Prop := sorry

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits in B is 486 -/
theorem sum_of_digits_B : 
  ∃ (B : ℕ), 
    (∃ (digits : Finset ℕ), 
      digits.card = digitsToRemove ∧ 
      B = removeDigits A digits ∧
      isSmallestPossible B A digits) →
    sumOfDigits B = 486 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_B_l1079_107955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_double_discount_l1079_107941

/-- Represents the true discount calculation --/
noncomputable def true_discount (present_value : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (present_value * rate * time) / 100

theorem double_time_double_discount 
  (amount : ℝ) (discount : ℝ) (time : ℝ) (rate : ℝ) 
  (h1 : amount > 0)
  (h2 : discount > 0)
  (h3 : time > 0)
  (h4 : rate > 0)
  (h5 : discount = true_discount (amount - discount) rate time) :
  2 * discount = true_discount (amount - discount) rate (2 * time) :=
by
  sorry

#check double_time_double_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_double_discount_l1079_107941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l1079_107992

/-- Represents the pizza and payment scenario --/
structure PizzaScenario where
  totalSlices : ℕ
  plainCost : ℚ
  anchovyCost : ℚ
  daveSlices : ℕ
  dougSlices : ℕ

/-- Calculates the payment difference between Dave and Doug --/
def paymentDifference (scenario : PizzaScenario) : ℚ :=
  let totalCost := scenario.plainCost + scenario.anchovyCost
  let daveCost := (scenario.daveSlices : ℚ) * totalCost / (scenario.totalSlices : ℚ)
  let dougCost := (scenario.dougSlices : ℚ) * scenario.plainCost / (scenario.totalSlices : ℚ)
  daveCost - dougCost

/-- The theorem to be proven --/
theorem pizza_payment_difference (scenario : PizzaScenario) :
  scenario.totalSlices = 8 ∧
  scenario.plainCost = 8 ∧
  scenario.anchovyCost = 2 ∧
  scenario.daveSlices = 5 ∧
  scenario.dougSlices = 3 →
  paymentDifference scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l1079_107992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_505th_term_l1079_107942

/-- Represents a sequence with alternating progression pattern -/
def AlternatingSequence (p q : ℝ) : ℕ → ℝ
  | 0 => p  -- Added case for 0
  | 1 => p
  | 2 => 9
  | 3 => 3*p - q
  | 4 => 3*p + q
  | 5 => 4*p
  | n + 6 => AlternatingSequence p q (n % 5 + 1)

/-- The theorem stating that the 505th term of the sequence is 20 -/
theorem alternating_sequence_505th_term (p q : ℝ) 
  (h1 : 9 - p = 3*p - q - 9)  -- Alternating pattern condition
  (h2 : q = 2)                -- Derived from solution, but could be a condition
  (h3 : p = 5)                -- Derived from solution, but could be a condition
  : AlternatingSequence p q 505 = 20 := by
  sorry

#eval AlternatingSequence 5 2 505  -- Added for demonstration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_505th_term_l1079_107942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l1079_107914

theorem simplify_and_rationalize :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17) = a * Real.sqrt b / 1309 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l1079_107914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l1079_107951

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (x - Real.pi / 12)

theorem problem :
  (f (Real.pi / 3) = 1) ∧
  (∀ θ : ℝ, 
    θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) → 
    Real.cos θ = 3 / 5 → 
    f (θ - Real.pi / 6) = -1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l1079_107951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l1079_107976

/-- The force equation for the airstream acting on a sail -/
noncomputable def force (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v)^2) / 2

/-- The instantaneous power equation -/
noncomputable def power (B S ρ v₀ v : ℝ) : ℝ :=
  force B S ρ v₀ v * v

/-- Theorem stating that the speed maximizing instantaneous power is one-third of the wind speed -/
theorem max_power_speed (B S ρ v₀ : ℝ) (hB : B > 0) (hS : S > 0) (hρ : ρ > 0) (hv₀ : v₀ > 0) :
  ∃ v : ℝ, v > 0 ∧ v = v₀ / 3 ∧ ∀ u : ℝ, u ≠ v → power B S ρ v₀ v ≥ power B S ρ v₀ u := by
  sorry

#check max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l1079_107976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_h_l1079_107983

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a / x + b

def h (x : ℝ) : ℝ := a * x^2 + b * x

theorem zeros_of_h (ha : a ≠ 0) (hf : f 1 = 0) :
  ∀ x, h x = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_h_l1079_107983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l1079_107900

/-- The equation of a hyperbola in the form (ax + b)^2/c^2 - (dy + e)^2/f^2 = 1 --/
structure Hyperbola where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  e : ℚ
  f : ℚ

/-- The center of a hyperbola --/
structure HyperbolaCenter where
  x : ℚ
  y : ℚ

/-- Function to calculate the center of a hyperbola --/
def findHyperbolaCenter (h : Hyperbola) : HyperbolaCenter :=
  { x := -h.b / h.a,
    y := h.e / h.d }

theorem hyperbola_center_correct (h : Hyperbola) 
    (hc : h.a = 4 ∧ h.b = 8 ∧ h.c = 8 ∧ h.d = 2 ∧ h.e = 6 ∧ h.f = 6) :
  findHyperbolaCenter h = { x := -2, y := 3 } := by
  sorry

#check hyperbola_center_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l1079_107900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1079_107915

/-- The area of a triangle PQR with vertices P(0, a), Q(b, 0), R(c, d) -/
noncomputable def triangleArea (a b c d : ℝ) : ℝ := (a * c + b * d - a * b) / 2

theorem triangle_area_formula 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_opposite : (c - b) * d > 0) : 
  triangleArea a b c d = |((c * a + d * b - a * b) / 2)| := by
  sorry

#check triangle_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1079_107915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1079_107968

def my_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 1 + 1 / a n

theorem a_5_value (a : ℕ → ℚ) (h1 : my_sequence a) (h2 : a 8 = 34/21) : 
  a 5 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1079_107968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_third_l1079_107954

noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 4 then
    (2 * x * y - x + 3) / (3 * x)
  else
    (x * y - y - 3) / (-3 * y)

theorem g_sum_equals_one_third :
  g 3 1 + g 3 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_third_l1079_107954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_exists_l1079_107928

-- Define the plane and points
variable {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P]
variable (A B C X : P)

-- Define the non-collinearity condition
def NonCollinear (A B C : P) : Prop :=
  ¬ (∃ (t : ℝ), C - A = t • (B - A))

-- Define the condition for point X
def SatisfiesCondition (X A B C : P) : Prop :=
  (‖X - A‖^2 + ‖X - B‖^2 + ‖A - B‖^2 = ‖X - B‖^2 + ‖X - C‖^2 + ‖B - C‖^2) ∧
  (‖X - B‖^2 + ‖X - C‖^2 + ‖B - C‖^2 = ‖X - C‖^2 + ‖X - A‖^2 + ‖C - A‖^2)

-- State the theorem
theorem unique_point_exists (hNC : NonCollinear A B C) :
  ∃! X, SatisfiesCondition X A B C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_exists_l1079_107928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_inequality_l1079_107904

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Statement of the theorem
theorem fractional_part_inequality :
  (∀ n : ℕ+, frac (n * Real.sqrt 3) > 1 / (n * Real.sqrt 3)) ∧
  (¬ ∃ c : ℝ, c > 1 ∧ ∀ n : ℕ+, frac (n * Real.sqrt 3) > c / (n * Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_inequality_l1079_107904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_margin_l1079_107993

/-- Represents the result of a mayoral election with six candidates. -/
structure ElectionResult where
  P : Nat
  Q : Nat
  R : Nat
  S : Nat
  T : Nat
  U : Nat
  M : Nat
  V : Nat
  total_votes : Nat

/-- Calculates the number of votes for a candidate given their percentage -/
def votes_for_candidate (percentage : Nat) (total_votes : Nat) : Nat :=
  (percentage * total_votes) / 100

/-- Theorem stating the relationship between the winning candidate's votes and the margin of victory -/
theorem winning_margin (result : ElectionResult)
    (h_total : result.total_votes = 55000)
    (h_p_max : result.P > result.Q ∧ result.P > result.R ∧ result.P > result.S ∧
               result.P > result.T ∧ result.P > result.U)
    (h_valid : ∀ x ∈ [result.P, result.Q, result.R, result.S, result.T, result.U],
               votes_for_candidate x result.total_votes ≥ result.V) :
  votes_for_candidate result.P result.total_votes -
  max (votes_for_candidate result.Q result.total_votes)
      (max (votes_for_candidate result.R result.total_votes)
           (max (votes_for_candidate result.S result.total_votes)
                (max (votes_for_candidate result.T result.total_votes)
                     (votes_for_candidate result.U result.total_votes)))) = result.M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_margin_l1079_107993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_implies_increasing_l1079_107910

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_decreasing_implies_increasing
  (f : ℝ → ℝ) (h_even : is_even_function f)
  (h_decreasing : decreasing_on f (Set.Ioi 0)) :
  increasing_on f (Set.Iio 0) :=
by
  sorry

#check even_decreasing_implies_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_implies_increasing_l1079_107910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l1079_107934

theorem ticket_price_possibilities : 
  let possible_prices := {x : ℕ | x > 0 ∧ 48 % x = 0 ∧ 64 % x = 0}
  Finset.card (Finset.filter (fun x => x ∈ possible_prices) (Finset.range 65)) = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l1079_107934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_isosceles_condition_l1079_107957

/-- Hyperbola S with given properties -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  eccentricity : ℝ
  min_distance : ℝ

/-- Line with given equation -/
def line (a b c : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 + b * p.2 + c = 0}

/-- Point belongs to hyperbola -/
def belongs_to_hyperbola (p : ℝ × ℝ) (S : Hyperbola) : Prop := 
  p.1^2 / 2 - p.2^2 = 1

/-- Theorem for the hyperbola equation and isosceles triangle condition -/
theorem hyperbola_and_isosceles_condition 
  (S : Hyperbola)
  (h_ecc : S.eccentricity = Real.sqrt 6 / 2)
  (h_dist : S.min_distance = 4 * Real.sqrt 3 / 3)
  (h_line : line (Real.sqrt 3) (-3) 5)
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h_AB : belongs_to_hyperbola A S ∧ belongs_to_hyperbola B S)
  (h_line_k : ∃ (m : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ A.2 = k * (A.1 + 2) ∧ B.2 = k * (B.1 + 2))
  (P : ℝ × ℝ := (0, 1))
  (h_isosceles : (A.1 - P.1)^2 + (A.2 - P.2)^2 = (B.1 - P.1)^2 + (B.2 - P.2)^2) :
  (∀ (x y : ℝ), belongs_to_hyperbola (x, y) S ↔ x^2 / 2 - y^2 = 1) ∧
  (k = 0 ∨ k = (-3 + Real.sqrt 11) / 2 ∨ k = (-3 - Real.sqrt 11) / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_isosceles_condition_l1079_107957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_eight_l1079_107909

def is_valid_digit_pair (a b : Nat) : Prop :=
  (a * 10 + b) % 19 = 0 ∨ (a * 10 + b) % 23 = 0

def valid_number (n : List Nat) : Prop :=
  n.length = 2022 ∧
  n.head? = some 4 ∧
  ∀ i, i < n.length - 1 → is_valid_digit_pair (n.get! i) (n.get! (i+1))

theorem last_digit_is_eight (n : List Nat) (h : valid_number n) :
  n.getLast? = some 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_eight_l1079_107909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potassium_bisulfate_formation_l1079_107966

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between Potassium hydroxide and Sulfuric acid -/
structure Reaction where
  koh : Moles  -- Potassium hydroxide
  h2so4 : Moles  -- Sulfuric acid
  khso4 : Moles  -- Potassium bisulfate
  h2o : Moles  -- Water

/-- The reaction satisfies the given conditions -/
def validReaction (r : Reaction) : Prop :=
  r.koh = (2 : ℝ) ∧ r.h2so4 = (2 : ℝ) ∧ r.h2so4 = r.khso4

theorem potassium_bisulfate_formation (r : Reaction) 
  (h : validReaction r) : r.khso4 = (2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potassium_bisulfate_formation_l1079_107966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1079_107949

/-- The function representing the left side of the inequality -/
def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

/-- The region in the first octant satisfying the inequality -/
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ f p.1 p.2.1 p.2.2 ≤ 8}

/-- The volume of the region S is 32/3 -/
theorem volume_of_region : MeasureTheory.volume S = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1079_107949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_over_129_l1079_107958

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def box1_capacity : ℕ := 4
def box2_capacity : ℕ := 5
def box3_capacity : ℕ := 6

def probability_same_box : ℚ :=
  let total_arrangements := Nat.choose total_textbooks box1_capacity *
                            Nat.choose (total_textbooks - box1_capacity) box2_capacity *
                            Nat.choose (total_textbooks - box1_capacity - box2_capacity) box3_capacity
  let favorable_outcomes := Nat.choose (total_textbooks - math_textbooks) 0 *
                            Nat.choose (total_textbooks - box1_capacity) box2_capacity +
                            Nat.choose (total_textbooks - math_textbooks) 1 *
                            Nat.choose (total_textbooks - box2_capacity) box1_capacity +
                            Nat.choose (total_textbooks - math_textbooks) 2 *
                            Nat.choose (total_textbooks - box3_capacity) box2_capacity
  (favorable_outcomes : ℚ) / total_arrangements

theorem probability_is_two_over_129 : probability_same_box = 2 / 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_over_129_l1079_107958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_x_intercept_of_given_line_l1079_107935

/-- The x-intercept of a line is the point where the line crosses the x-axis (i.e., where y = 0) -/
noncomputable def x_intercept (a b c : ℝ) : ℝ := c / a

/-- A point (x, y) lies on a line ax + by = c if and only if ax + by = c -/
def on_line (x y a b c : ℝ) : Prop := a * x + b * y = c

theorem x_intercept_of_line (a b c : ℝ) (h₁ : a ≠ 0) :
  on_line (x_intercept a b c) 0 a b c := by sorry

theorem x_intercept_of_given_line :
  x_intercept 4 7 28 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_x_intercept_of_given_line_l1079_107935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firm_ratio_change_l1079_107902

/-- Represents the ratio of partners to associates in a firm -/
structure FirmRatio where
  partners : ℕ
  associates : ℕ
  deriving Repr

/-- Calculates the new ratio after hiring more associates -/
def newRatio (initial : FirmRatio) (currentPartners : ℕ) (newHires : ℕ) : FirmRatio :=
  let initialAssociates := (initial.associates * currentPartners) / initial.partners
  let newAssociates := initialAssociates + newHires
  let gcd := Nat.gcd currentPartners newAssociates
  { partners := currentPartners / gcd, associates := newAssociates / gcd }

theorem firm_ratio_change (initial : FirmRatio) (currentPartners newHires : ℕ) :
  initial.partners = 2 ∧ 
  initial.associates = 63 ∧ 
  currentPartners = 20 ∧ 
  newHires = 50 →
  newRatio initial currentPartners newHires = { partners := 1, associates := 34 } :=
by
  sorry

#eval newRatio { partners := 2, associates := 63 } 20 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firm_ratio_change_l1079_107902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_ratio_l1079_107921

/-- Represents a configuration of a spherical cap and a cylinder sharing an inscribed sphere -/
structure SphereCylinderConfig where
  R : ℝ  -- Radius of the inscribed sphere
  (R_pos : R > 0)

/-- Volume of the spherical cap -/
noncomputable def spherical_cap_volume (config : SphereCylinderConfig) : ℝ :=
  4 / 3 * Real.pi * config.R ^ 3

/-- Volume of the cylinder -/
noncomputable def cylinder_volume (config : SphereCylinderConfig) : ℝ :=
  2 * Real.pi * config.R ^ 3

/-- Ratio of spherical cap volume to cylinder volume -/
noncomputable def volume_ratio (config : SphereCylinderConfig) : ℝ :=
  spherical_cap_volume config / cylinder_volume config

theorem min_volume_ratio :
  ∀ config : SphereCylinderConfig, volume_ratio config ≥ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_ratio_l1079_107921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divisors_congruence_l1079_107987

theorem cube_divisors_congruence (x d : ℕ) (h1 : ∃ n : ℕ, x = n^3 ∧ n > 0) 
  (h2 : d = (Finset.filter (· ∣ x) (Finset.range (x + 1))).card) : 
  d % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divisors_congruence_l1079_107987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_coordinates_l1079_107908

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- A point is a symmetry center of a function if the function is symmetric about that point. -/
def IsSymmetryCenter (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (c.1 + x) = f (c.1 - x)

theorem symmetry_center_coordinates
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_period : ∀ x : ℝ, f ω φ (x + 4 * π) = f ω φ x)
  (h_max : ∀ x : ℝ, f ω φ x ≤ f ω φ (π / 3)) :
  ∃ c : ℝ × ℝ, c = (-π / 3, 0) ∧ IsSymmetryCenter (f ω φ) c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_coordinates_l1079_107908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_undefined_at_270_degrees_l1079_107996

/-- If the terminal side of angle α passes through point P(0, -4), then tan α does not exist. -/
theorem tan_undefined_at_270_degrees (α : ℝ) : 
  (∃ k : ℤ, α = 2 * Real.pi * k + 3 * Real.pi / 2) → ¬ ∃ (x : ℝ), Real.tan α = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_undefined_at_270_degrees_l1079_107996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_approximation_l1079_107944

/-- Given a principal amount where the compound interest for 4 years at 10% per annum
    compounded annually is 993, prove that the simple interest for the same period
    is approximately 856.19 -/
theorem simple_interest_approximation (P : ℝ) : 
  P * (1 + 0.1)^4 - P = 993 →
  abs (P * 0.1 * 4 - 856.19) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_approximation_l1079_107944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_distance_correct_l1079_107936

/-- Given an equilateral triangle ABC with area S and a smaller triangle A₁B₁C₁ with area Q
    formed by lines parallel to ABC's sides at equal distances, this function calculates
    the distance between the parallel sides of ABC and A₁B₁C₁. -/
noncomputable def parallelSidesDistance (S Q : ℝ) : Set ℝ :=
  if Q ≥ 1/4 * S then
    {Real.sqrt 3 / 3 * (Real.sqrt S - Real.sqrt Q)}
  else
    {Real.sqrt 3 / 3 * (Real.sqrt S - Real.sqrt Q),
     Real.sqrt 3 / 3 * (Real.sqrt S + Real.sqrt Q)}

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a triangle in a plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateralTriangle (t : Triangle) : Prop := sorry

/-- Function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Predicate to check if two triangles are parallel -/
def IsParallelTriangle (t1 t2 : Triangle) : Prop := sorry

/-- Function to calculate the distance between two lines -/
noncomputable def distance (p1 p2 : Point) (q1 q2 : Point) : ℝ := sorry

theorem parallel_sides_distance_correct (S Q : ℝ) (hS : S > 0) (hQ : Q > 0) :
  ∀ d ∈ parallelSidesDistance S Q,
    ∃ (t1 t2 : Triangle),
      IsEquilateralTriangle t1 ∧
      area t1 = S ∧
      IsParallelTriangle t1 t2 ∧
      area t2 = Q ∧
      d = distance t1.A t1.B t2.A t2.B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_distance_correct_l1079_107936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_square_areas_l1079_107920

theorem max_sum_square_areas (A B C : EuclideanSpace ℝ (Fin 2)) : 
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  AB = 2 ∧ AC = 3 →
  (AB^2 + BC^2 + AC^2) ≤ 9 ∧ ∃ (A' B' C' : EuclideanSpace ℝ (Fin 2)), 
    dist A' B'^2 + dist B' C'^2 + dist A' C'^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_square_areas_l1079_107920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_satisfies_differential_equation_l1079_107930

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := Real.cos (2 * x)

-- State the theorem
theorem cos_2x_satisfies_differential_equation :
  ∀ x : ℝ, (deriv (deriv y) x) + 4 * (y x) = 0 := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_satisfies_differential_equation_l1079_107930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_smallest_angle_is_smallest_l1079_107933

/-- The smallest positive angle θ in degrees that satisfies the given equation -/
def smallest_angle : ℝ := 21

theorem smallest_angle_satisfies_equation :
  Real.cos (smallest_angle * Real.pi / 180) = 
    Real.sin (45 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) - 
    Real.sin (30 * Real.pi / 180) - Real.cos (24 * Real.pi / 180) :=
by sorry

theorem smallest_angle_is_smallest (θ : ℝ) :
  θ > 0 ∧
  Real.cos (θ * Real.pi / 180) = 
    Real.sin (45 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) - 
    Real.sin (30 * Real.pi / 180) - Real.cos (24 * Real.pi / 180) →
  θ ≥ smallest_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_smallest_angle_is_smallest_l1079_107933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1079_107986

theorem calculate_expression : ((-2023 : ℝ) ^ 0) + |(-Real.sqrt 2)| - 2 * Real.cos (π / 4) - (216 : ℝ) ^ (1/3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1079_107986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1079_107998

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : y - Real.cos z / x ≠ 0) : 
  (x - Real.sin z / y) / (y - Real.cos z / x) = (x^2 * y - x * Real.sin z) / (x * y^2 - y * Real.cos z) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1079_107998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_convergence_l1079_107962

/-- The infinite series ∑_{n=1}^∞ (n^3 + n^2 - n) / ((n + 3)!) converges to 1/2 -/
theorem infinite_series_convergence :
  ∑' n : ℕ, (n^3 + n^2 - n : ℝ) / (Nat.factorial (n + 3)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_convergence_l1079_107962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l1079_107985

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define a line with slope -2 and y-intercept b
def my_line (x y b : ℝ) : Prop := 2*x + y + b = 0

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop :=
  ∃ x y, my_circle x y ∧ my_line x y b

-- Define the theorem
theorem tangent_lines :
  (∀ b, is_tangent b ↔ (b = 3 ∨ b = -7)) := by
  sorry

#check tangent_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l1079_107985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1079_107911

/-- The volume of a pyramid with a rectangular base and equal side edges --/
theorem pyramid_volume (base_length base_width side_edge : ℝ) 
  (h_base_length : base_length = 4)
  (h_base_width : base_width = 10)
  (h_side_edge : side_edge = 15) :
  (1/3) * (base_length * base_width) * 
    Real.sqrt (side_edge^2 - ((Real.sqrt (base_length^2 + base_width^2))/2)^2) = 560/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1079_107911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_is_96_l1079_107982

-- Define the rectangular solid
structure RectangularSolid where
  a : ℝ  -- middle term of the geometric progression
  r : ℝ  -- common ratio
  volume : ℝ
  surfaceArea : ℝ

-- Define the conditions
def validSolid (s : RectangularSolid) : Prop :=
  s.volume = 216 ∧
  s.surfaceArea = 288 ∧
  s.r > 0 ∧
  s.a > 0

-- Define the sum of edge lengths
noncomputable def sumOfEdges (s : RectangularSolid) : ℝ :=
  4 * (s.a / s.r + s.a + s.a * s.r)

-- Theorem statement
theorem sum_of_edges_is_96 (s : RectangularSolid) 
  (h : validSolid s) : sumOfEdges s = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_is_96_l1079_107982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1079_107907

/-- Calculates the principal given simple interest, rate, and time -/
noncomputable def calculate_principal (simple_interest rate time : ℝ) : ℝ :=
  simple_interest * 100 / (rate * time)

theorem principal_calculation (simple_interest rate time : ℝ) 
  (h1 : simple_interest = 6016.75)
  (h2 : rate = 8)
  (h3 : time = 5) :
  calculate_principal simple_interest rate time = 15041.875 := by
  -- Unfold the definition of calculate_principal
  unfold calculate_principal
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1079_107907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1079_107948

open Real

/-- The function f(x) defined on the positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1/x - a * log x

/-- The derivative of f(x) with respect to x. -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 + 1/x^2 - a/x

/-- Condition for x to be an extreme point of f. -/
def is_extreme_point (a : ℝ) (x : ℝ) : Prop := f_deriv a x = 0

/-- The slope of the line passing through two points on the graph of f. -/
noncomputable def k (a : ℝ) (x₁ x₂ : ℝ) : ℝ := (f a x₁ - f a x₂) / (x₁ - x₂)

/-- Main theorem: There does not exist an 'a' such that the slope between two extreme points equals 2 - a. -/
theorem no_solution_exists : 
  ¬ ∃ (a : ℝ), ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    is_extreme_point a x₁ ∧ 
    is_extreme_point a x₂ ∧ 
    k a x₁ x₂ = 2 - a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1079_107948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_point_inside_iff_odd_marked_vertices_l1079_107963

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Define a line as a pair of points in 2D space
def Line := (ℝ × ℝ) × (ℝ × ℝ)

-- Function to check if a point is on a line
def isPointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

-- Function to check if a point is inside a polygon
def isPointInside (p : ℝ × ℝ) (poly : Polygon) : Prop := sorry

-- Function to check if two points are on opposite sides of a third point on a line
def areOpposite (p1 p2 p : ℝ × ℝ) (l : Line) : Prop := sorry

-- Function to count marked vertices on each side of the line
def countMarkedVertices (poly : Polygon) (l : Line) (p : ℝ × ℝ) : ℕ × ℕ := sorry

-- Function to get edges of a polygon
def edges (poly : Polygon) : List Line := sorry

-- Main theorem
theorem polygon_point_inside_iff_odd_marked_vertices 
  (poly : Polygon) (l : Line) (p : ℝ × ℝ) 
  (h1 : isPointOnLine p l) 
  (h2 : ∀ (e : Line), e ∈ (edges poly) → ∃! q, q ≠ p ∧ isPointOnLine q l ∧ isPointOnLine q e) :
  isPointInside p poly ↔ 
    let (left, right) := countMarkedVertices poly l p
    Odd left ∧ Odd right :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_point_inside_iff_odd_marked_vertices_l1079_107963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l1079_107917

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

-- Define the points
def pointA : ℝ × ℝ := (0, -1/2)
def pointB : ℝ × ℝ := (-3/2, -1)
def pointC : ℝ × ℝ := (1, 0)
def pointD : ℝ × ℝ := (-2, 3)
def pointE : ℝ × ℝ := (2, 1/4)

-- Theorem statement
theorem point_not_on_graph :
  (f pointA.1 = pointA.2) ∧
  (f pointC.1 = pointC.2) ∧
  (f pointE.1 = pointE.2) ∧
  (f pointB.1 ≠ pointB.2) ∧
  ¬(∃ y : ℝ, f pointD.1 = y ∧ y = pointD.2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l1079_107917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_terms_formula_l1079_107913

/-- Given a geometric sequence {a_n} where a_n = 2 * 3^(n-1),
    this function computes the sum of the first n terms of a new sequence
    formed by the even terms of the original sequence. -/
def sum_of_even_terms (n : ℕ) : ℚ :=
  (3 / 4) * (9^n - 1)

/-- Theorem stating that the sum of the first n terms of the new sequence
    formed by the even terms of the original geometric sequence
    is equal to (3/4) * (9^n - 1). -/
theorem sum_of_even_terms_formula (n : ℕ) :
  let a : ℕ → ℚ := fun k => 2 * 3^(k - 1)
  let even_terms : ℕ → ℚ := fun k => a (2 * k)
  (Finset.range n).sum even_terms = sum_of_even_terms n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_terms_formula_l1079_107913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_fourth_rods_count_l1079_107906

def rod_lengths : List ℕ := List.range 41

def given_rods : List ℕ := [5, 10, 20]

def is_valid_fourth_rod (a b c d : ℕ) : Bool :=
  a + b + c > d && a + b + d > c && a + c + d > b && b + c + d > a

def count_valid_fourth_rods (given : List ℕ) (all : List ℕ) : ℕ :=
  (all.filter (fun d => 
    !given.contains d && 
    is_valid_fourth_rod given[0]! given[1]! given[2]! d
  )).length

theorem valid_fourth_rods_count :
  count_valid_fourth_rods given_rods rod_lengths = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_fourth_rods_count_l1079_107906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_in_bag_l1079_107994

/-- Represents the number of cups of food in a bag -/
noncomputable def cups_per_bag : ℚ := 7/2

/-- Represents the cost of the puppy in dollars -/
def puppy_cost : ℚ := 10

/-- Represents the number of weeks for which food is bought -/
def weeks_of_food : ℕ := 3

/-- Represents the amount of food the puppy eats per day in cups -/
def food_per_day : ℚ := 1/3

/-- Represents the cost of a bag of food in dollars -/
def bag_cost : ℚ := 2

/-- Represents the total cost in dollars -/
def total_cost : ℚ := 14

/-- Proves that the number of cups of food in a bag is 3.5 -/
theorem cups_in_bag :
  cups_per_bag = (total_cost - puppy_cost) / bag_cost * (7 * weeks_of_food * food_per_day) / (7 * weeks_of_food * food_per_day / bag_cost) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_in_bag_l1079_107994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_zeros_of_F_l1079_107978

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 + x - x^2/2 + x^3/3
noncomputable def g (x : ℝ) : ℝ := 1 - x + x^2/2 - x^3/3

-- Define the product function F
noncomputable def F (x : ℝ) : ℝ := f x * g x

-- Theorem statement
theorem min_interval_for_zeros_of_F :
  ∃ (a b : ℤ), a < b ∧
  (∀ x : ℝ, F x = 0 → a ≤ x ∧ x ≤ b) ∧
  (∀ c d : ℤ, c < d → (∀ x : ℝ, F x = 0 → c ≤ x ∧ x ≤ d) → d - c ≥ b - a) ∧
  b - a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_zeros_of_F_l1079_107978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l1079_107953

theorem first_six_average (numbers : Fin 11 → ℝ) 
  (h_total_avg : (Finset.sum (Finset.range 11) (λ i => numbers i)) / 11 = 60)
  (h_last_six_avg : (Finset.sum (Finset.range 6) (λ i => numbers (i + 5))) / 6 = 65)
  (h_sixth : numbers 5 = 258) :
  (Finset.sum (Finset.range 6) (λ i => numbers i)) / 6 = 88 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l1079_107953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_triangle_l1079_107939

/-- The diameter of the inscribed circle in a triangle with side lengths a, b, and c -/
noncomputable def inscribedCircleDiameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s

/-- Theorem: The diameter of the inscribed circle in triangle PQR with side lengths 13, 8, and 15 is 10√3/3 -/
theorem inscribed_circle_diameter_specific_triangle :
  inscribedCircleDiameter 13 8 15 = 10 * Real.sqrt 3 / 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_triangle_l1079_107939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_is_60_minutes_l1079_107927

/-- Represents the rate of work in project completion per hour -/
structure WorkRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a day's work schedule -/
structure WorkDay where
  duration : ℝ
  completion : ℝ
  workers : List WorkRate

/-- The lunch break duration in hours -/
noncomputable def lunch_break : ℝ := 1

/-- Ray's work rate -/
noncomputable def ray : WorkRate where
  rate := 0.1 / (3 - lunch_break)
  rate_pos := by
    sorry -- Proof that the rate is positive

/-- Combined work rate of the two assistants -/
noncomputable def assistants : WorkRate where
  rate := 0.3 / (7 - lunch_break)
  rate_pos := by
    sorry -- Proof that the rate is positive

/-- First day's work schedule -/
noncomputable def day1 : WorkDay :=
  { duration := 9 - lunch_break
    completion := 0.6
    workers := [ray, assistants] }

/-- Second day's work schedule -/
noncomputable def day2 : WorkDay :=
  { duration := 7 - lunch_break
    completion := 0.3
    workers := [assistants] }

/-- Third day's work schedule -/
noncomputable def day3 : WorkDay :=
  { duration := 3 - lunch_break
    completion := 0.1
    workers := [ray] }

/-- Theorem stating that the lunch break is 60 minutes -/
theorem lunch_break_is_60_minutes :
  lunch_break * 60 = 60 := by
  sorry -- Proof that lunch_break * 60 = 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_is_60_minutes_l1079_107927
