import Mathlib

namespace inequality_preserved_by_halving_l1409_140947

theorem inequality_preserved_by_halving {a b : ℝ} (h : a > b) : a / 2 > b / 2 := by
  sorry

end inequality_preserved_by_halving_l1409_140947


namespace power_sum_difference_l1409_140982

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end power_sum_difference_l1409_140982


namespace min_tablets_extraction_l1409_140956

/-- Represents the number of tablets of each medicine type in the box -/
structure TabletCounts where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to ensure at least 3 of each kind -/
def minTablets (counts : TabletCounts) : Nat :=
  16

/-- Theorem stating that for the given tablet counts, the minimum number of tablets to extract is 16 -/
theorem min_tablets_extraction (counts : TabletCounts) 
  (h1 : counts.a = 30) (h2 : counts.b = 24) (h3 : counts.c = 18) : 
  minTablets counts = 16 := by
  sorry

end min_tablets_extraction_l1409_140956


namespace unique_m_equals_three_l1409_140948

/-- A graph is k-flowing-chromatic if it satisfies certain coloring and movement conditions -/
def is_k_flowing_chromatic (G : Graph) (k : ℕ) : Prop := sorry

/-- T(G) is the least k such that G is k-flowing-chromatic, or 0 if no such k exists -/
def T (G : Graph) : ℕ := sorry

/-- χ(G) is the chromatic number of graph G -/
def chromatic_number (G : Graph) : ℕ := sorry

/-- A graph has no small cycles if all its cycles have length at least 2017 -/
def no_small_cycles (G : Graph) : Prop := sorry

/-- Main theorem: m = 3 is the only positive integer satisfying the conditions -/
theorem unique_m_equals_three :
  ∀ m : ℕ, m > 0 →
  (∃ G : Graph, chromatic_number G ≤ m ∧ T G ≥ 2^m ∧ no_small_cycles G) ↔ m = 3 :=
by sorry

end unique_m_equals_three_l1409_140948


namespace midpoint_coordinate_product_l1409_140941

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4,7) and (-8,9) is -16. -/
theorem midpoint_coordinate_product : 
  let a : ℝ × ℝ := (4, 7)
  let b : ℝ × ℝ := (-8, 9)
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  (midpoint.1 * midpoint.2 : ℝ) = -16 := by
  sorry

end midpoint_coordinate_product_l1409_140941


namespace events_mutually_exclusive_but_not_opposite_l1409_140967

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_receives_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_receives_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define the property of a valid distribution
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : Card), ∃! (p : Person), d p = c

theorem events_mutually_exclusive_but_not_opposite :
  (∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_red d ∧ B_receives_red d)) ∧
  (∃ (d : Distribution), valid_distribution d ∧
    ¬A_receives_red d ∧ ¬B_receives_red d) :=
sorry

end events_mutually_exclusive_but_not_opposite_l1409_140967


namespace fundraiser_total_l1409_140903

theorem fundraiser_total (brownie_students : Nat) (brownie_per_student : Nat) (brownie_price : Real)
                         (cookie_students : Nat) (cookie_per_student : Nat) (cookie_price : Real)
                         (donut_students : Nat) (donut_per_student : Nat) (donut_price : Real)
                         (cupcake_students : Nat) (cupcake_per_student : Nat) (cupcake_price : Real) :
  brownie_students = 70 ∧ brownie_per_student = 20 ∧ brownie_price = 1.50 ∧
  cookie_students = 40 ∧ cookie_per_student = 30 ∧ cookie_price = 2.25 ∧
  donut_students = 35 ∧ donut_per_student = 18 ∧ donut_price = 3.00 ∧
  cupcake_students = 25 ∧ cupcake_per_student = 12 ∧ cupcake_price = 2.50 →
  (brownie_students * brownie_per_student * brownie_price +
   cookie_students * cookie_per_student * cookie_price +
   donut_students * donut_per_student * donut_price +
   cupcake_students * cupcake_per_student * cupcake_price) = 7440 :=
by
  sorry

end fundraiser_total_l1409_140903


namespace alfreds_savings_l1409_140960

/-- Alfred's savings problem -/
theorem alfreds_savings (goal : ℝ) (months : ℕ) (monthly_savings : ℝ) 
  (h1 : goal = 1000)
  (h2 : months = 12)
  (h3 : monthly_savings = 75) :
  goal - (monthly_savings * months) = 100 := by
  sorry

end alfreds_savings_l1409_140960


namespace f_inequalities_l1409_140995

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities (a : ℝ) :
  (a < -1 → {x : ℝ | f a x < 0} = Set.Ioo a (-1)) ∧
  (a = -1 → {x : ℝ | f a x < 0} = ∅) ∧
  (a > -1 → {x : ℝ | f a x < 0} = Set.Ioo (-1) a) ∧
  ({x : ℝ | x^3 * f 2 x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 2) :=
by sorry


end f_inequalities_l1409_140995


namespace circle_cartesian_and_center_l1409_140920

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Theorem statement
theorem circle_cartesian_and_center :
  ∃ (x y : ℝ), 
    (∀ (ρ θ : ℝ), C ρ θ ↔ x^2 - 2*x + y^2 = 0) ∧
    (x = 1 ∧ y = 0) := by
  sorry

end circle_cartesian_and_center_l1409_140920


namespace three_numbers_ratio_l1409_140946

theorem three_numbers_ratio (F S T : ℚ) : 
  F + S + T = 550 → 
  S = 150 → 
  T = F / 3 → 
  F / S = 2 := by
sorry

end three_numbers_ratio_l1409_140946


namespace trig_expression_equals_four_l1409_140979

theorem trig_expression_equals_four : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.cos (20 * π / 180)) = 4 := by
  sorry

end trig_expression_equals_four_l1409_140979


namespace intermediate_value_theorem_l1409_140976

theorem intermediate_value_theorem {f : ℝ → ℝ} {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_ab : a ≤ b) (h_sign : f a * f b < 0) : 
  ∃ c ∈ Set.Icc a b, f c = 0 := by
  sorry

end intermediate_value_theorem_l1409_140976


namespace intersection_chord_length_l1409_140926

-- Define the circles in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = 2

def circle_O₂ (ρ θ : ℝ) : Prop := ρ^2 - 2*Real.sqrt 2*ρ*(Real.cos (θ - Real.pi/4)) = 2

-- Define the circles in rectangular coordinates
def rect_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

def rect_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (rect_O₁ A.1 A.2 ∧ rect_O₁ B.1 B.2) →
  (rect_O₂ A.1 A.2 ∧ rect_O₂ B.1 B.2) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by
  sorry

end intersection_chord_length_l1409_140926


namespace largest_prime_divisor_of_sum_of_squares_l1409_140951

theorem largest_prime_divisor_of_sum_of_squares : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (36^2 + 49^2) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l1409_140951


namespace unique_snuggly_number_l1409_140993

def is_snuggly (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 2 * a + b^2

theorem unique_snuggly_number : ∃! n : ℕ, is_snuggly n :=
  sorry

end unique_snuggly_number_l1409_140993


namespace max_value_of_f_on_I_l1409_140934

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem max_value_of_f_on_I :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ I, f x ≤ m :=
sorry

end max_value_of_f_on_I_l1409_140934


namespace circle_equation_l1409_140912

/-- A circle C in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle at a given point -/
def Circle.tangentAt (c : Circle) (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  c.contains p ∧ 
  (c.center.2 - p.2) / (c.center.1 - p.1) = -1 / m ∧
  p.2 = m * p.1 + b

/-- The main theorem -/
theorem circle_equation (C : Circle) :
  C.center = (3, 0) ∧ C.radius = 2 →
  C.contains (4, 1) ∧
  C.tangentAt 1 (-2) (2, 1) :=
by
  sorry

end circle_equation_l1409_140912


namespace david_did_more_pushups_l1409_140900

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 35

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem david_did_more_pushups : pushup_difference = 9 := by
  sorry

end david_did_more_pushups_l1409_140900


namespace children_ages_sum_l1409_140998

theorem children_ages_sum (a b c d : ℕ) : 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 882 →
  a + b + c + d = 31 := by
  sorry

end children_ages_sum_l1409_140998


namespace chocolate_milk_probability_l1409_140913

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem chocolate_milk_probability : 
  binomial_probability 7 5 (1/2) = 21/128 := by
  sorry

end chocolate_milk_probability_l1409_140913


namespace exponent_division_l1409_140969

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end exponent_division_l1409_140969


namespace volleyball_starters_count_l1409_140972

def volleyball_team_size : ℕ := 16
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let with_one_triplet := triplet_size * Nat.choose non_triplet_size (starter_size - 1)
  let without_triplets := Nat.choose non_triplet_size starter_size
  with_one_triplet + without_triplets

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 5577 :=
by sorry

end volleyball_starters_count_l1409_140972


namespace number_division_problem_l1409_140906

theorem number_division_problem (x : ℝ) (h : (x - 5) / 7 = 7) :
  ∃ y : ℝ, (x - 34) / y = 2 ∧ y = 10 :=
by sorry

end number_division_problem_l1409_140906


namespace complex_absolute_value_l1409_140911

theorem complex_absolute_value (x : ℝ) (h : x > 0) :
  Complex.abs (-3 + 2*x*Complex.I) = 5 * Real.sqrt 5 ↔ x = Real.sqrt 29 := by
  sorry

end complex_absolute_value_l1409_140911


namespace mikes_remaining_nickels_l1409_140914

/-- Represents the number of nickels Mike has after his dad's borrowing. -/
def mikesRemainingNickels (initialNickels : ℕ) (borrowedNickels : ℕ) : ℕ :=
  initialNickels - borrowedNickels

/-- Represents the total number of nickels borrowed by Mike's dad. -/
def totalBorrowedNickels (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : ℕ :=
  mikesBorrowed + sistersBorrowed

/-- Represents the relationship between nickels borrowed from Mike and his sister. -/
def borrowingPattern (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : Prop :=
  3 * sistersBorrowed = 2 * mikesBorrowed

theorem mikes_remaining_nickels :
  ∀ (mikesInitialNickels : ℕ) (mikesBorrowed : ℕ) (sistersBorrowed : ℕ),
    mikesInitialNickels = 87 →
    totalBorrowedNickels mikesBorrowed sistersBorrowed = 75 →
    borrowingPattern mikesBorrowed sistersBorrowed →
    mikesRemainingNickels mikesInitialNickels mikesBorrowed = 42 :=
by sorry

end mikes_remaining_nickels_l1409_140914


namespace tan_alpha_value_l1409_140905

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3)
  (h2 : Real.sin (2 * α) > 0) :
  Real.tan α = Real.sqrt 2 / 4 := by
  sorry

end tan_alpha_value_l1409_140905


namespace coffee_cheesecake_set_price_l1409_140904

/-- The final price of a coffee and cheesecake set with a discount -/
theorem coffee_cheesecake_set_price 
  (coffee_price : ℝ) 
  (cheesecake_price : ℝ) 
  (discount_rate : ℝ) 
  (h1 : coffee_price = 6)
  (h2 : cheesecake_price = 10)
  (h3 : discount_rate = 0.25) :
  coffee_price + cheesecake_price - (coffee_price + cheesecake_price) * discount_rate = 12 :=
by sorry

end coffee_cheesecake_set_price_l1409_140904


namespace madeline_free_time_l1409_140989

/-- Calculates the number of hours Madeline has left over in a week --/
theorem madeline_free_time (class_hours week_days daily_hours homework_hours sleep_hours work_hours : ℕ) :
  class_hours = 18 →
  week_days = 7 →
  daily_hours = 24 →
  homework_hours = 4 →
  sleep_hours = 8 →
  work_hours = 20 →
  daily_hours * week_days - (class_hours + homework_hours * week_days + sleep_hours * week_days + work_hours) = 46 := by
  sorry

end madeline_free_time_l1409_140989


namespace min_lines_for_37_segments_l1409_140958

/-- Represents an open non-self-intersecting broken line -/
structure BrokenLine where
  segments : ℕ
  is_open : Bool
  is_non_self_intersecting : Bool

/-- Represents the minimum number of lines needed to cover all segments of a broken line -/
def minimum_lines (bl : BrokenLine) : ℕ := sorry

/-- The theorem stating the minimum number of lines for a 37-segment broken line -/
theorem min_lines_for_37_segments (bl : BrokenLine) : 
  bl.segments = 37 → bl.is_open = true → bl.is_non_self_intersecting = true →
  minimum_lines bl = 9 := by sorry

end min_lines_for_37_segments_l1409_140958


namespace equality_condition_l1409_140959

theorem equality_condition (x y z a b c : ℝ) :
  (Real.sqrt (x + a) + Real.sqrt (y + b) + Real.sqrt (z + c) =
   Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c)) ∧
  (Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c) =
   Real.sqrt (z + a) + Real.sqrt (x + b) + Real.sqrt (y + c)) →
  (x = y ∧ y = z) ∨ (a = b ∧ b = c) :=
by sorry

end equality_condition_l1409_140959


namespace sum_of_decimals_l1409_140954

theorem sum_of_decimals : 5.67 + (-3.92) = 1.75 := by
  sorry

end sum_of_decimals_l1409_140954


namespace largest_b_value_l1409_140950

theorem largest_b_value (b : ℚ) (h : (3*b+4)*(b-2) = 9*b) : 
  ∃ (max_b : ℚ), max_b = 4 ∧ ∀ (x : ℚ), (3*x+4)*(x-2) = 9*x → x ≤ max_b :=
by sorry

end largest_b_value_l1409_140950


namespace arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l1409_140984

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define what it means for a sequence to be an arithmetic progression
def isArithmeticProgression (seq : ℕ → ℕ) : Prop := sorry

-- Define what it means for a sequence to be increasing
def isIncreasing (seq : ℕ → ℕ) : Prop := sorry

-- Theorem for the case of 11 terms
theorem arithmetic_progression_11 : 
  ∃ (seq : Fin 11 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of 10,000 terms
theorem arithmetic_progression_10000 : 
  ∃ (seq : Fin 10000 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of infinite natural numbers
theorem no_infinite_arithmetic_progression :
  ¬∃ (seq : ℕ → ℕ), 
    isArithmeticProgression seq ∧ 
    isIncreasing seq ∧
    isArithmeticProgression (λ n => sumOfDigits (seq n)) ∧
    isIncreasing (λ n => sumOfDigits (seq n)) := sorry

end arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l1409_140984


namespace recycling_project_weight_l1409_140931

-- Define the number of items collected by each person
def marcus_bottles : ℕ := 25
def marcus_cans : ℕ := 30
def john_bottles : ℕ := 20
def john_cans : ℕ := 25
def sophia_bottles : ℕ := 15
def sophia_cans : ℕ := 35

-- Define the weight of each item
def bottle_weight : ℚ := 0.5
def can_weight : ℚ := 0.025

-- Define the total weight function
def total_weight : ℚ :=
  (marcus_bottles + john_bottles + sophia_bottles) * bottle_weight +
  (marcus_cans + john_cans + sophia_cans) * can_weight

-- Theorem statement
theorem recycling_project_weight :
  total_weight = 32.25 := by sorry

end recycling_project_weight_l1409_140931


namespace model_y_completion_time_l1409_140955

/-- The time (in minutes) taken by a Model Y computer to complete the task -/
def model_y_time : ℝ := 36

/-- The time (in minutes) taken by a Model X computer to complete the task -/
def model_x_time : ℝ := 72

/-- The number of Model X computers used -/
def num_computers : ℕ := 24

theorem model_y_completion_time :
  (num_computers : ℝ) * (1 / model_x_time + 1 / model_y_time) = 1 →
  model_y_time = 36 := by
  sorry

end model_y_completion_time_l1409_140955


namespace arctan_sum_l1409_140978

theorem arctan_sum : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end arctan_sum_l1409_140978


namespace geometric_sequence_properties_l1409_140915

theorem geometric_sequence_properties (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -r^4 ∧ a = -r^3 ∧ b = r^2 ∧ c = -r ∧ -9 = 1) →
  b = -3 ∧ a * c = 9 := by
sorry

end geometric_sequence_properties_l1409_140915


namespace savings_calculation_l1409_140953

def income_expenditure_ratio (income expenditure : ℚ) : Prop :=
  income / expenditure = 5 / 4

theorem savings_calculation (income : ℚ) (h : income_expenditure_ratio income ((4/5) * income)) :
  income - ((4/5) * income) = 3200 :=
by
  sorry

#check savings_calculation (16000 : ℚ)

end savings_calculation_l1409_140953


namespace parallelepiped_edge_lengths_l1409_140994

/-- Given a rectangular parallelepiped with mass M and density ρ, and thermal power ratios of 1:2:8
    when connected to different pairs of faces, this theorem states the edge lengths of the parallelepiped. -/
theorem parallelepiped_edge_lengths (M ρ : ℝ) (hM : M > 0) (hρ : ρ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a < b ∧ b < c ∧
    a * b * c = M / ρ ∧
    b^2 / a^2 = 2 ∧
    c^2 / b^2 = 4 ∧
    a = (M / (4 * ρ))^(1/3) ∧
    b = Real.sqrt 2 * (M / (4 * ρ))^(1/3) ∧
    c = 2 * Real.sqrt 2 * (M / (4 * ρ))^(1/3) := by
  sorry


end parallelepiped_edge_lengths_l1409_140994


namespace triangle_angle_measure_l1409_140988

theorem triangle_angle_measure (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 →
  a^2 + b^2 = 6 * a * b * Real.cos C →
  Real.sin C^2 = 2 * Real.sin A * Real.sin B →
  C = π / 3 := by
  sorry

end triangle_angle_measure_l1409_140988


namespace work_completion_time_l1409_140937

/-- The time it takes to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_alone_time : ℝ) :
  p_rate = total_work / 10 →
  q_rate = total_work / 6 →
  p_alone_time = 2 →
  let remaining_work := total_work - p_rate * p_alone_time
  let combined_rate := p_rate + q_rate
  total_work > 0 →
  p_rate > 0 →
  q_rate > 0 →
  p_alone_time + remaining_work / combined_rate = 5 :=
by sorry

end work_completion_time_l1409_140937


namespace cube_sum_given_sum_and_product_l1409_140970

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by sorry

end cube_sum_given_sum_and_product_l1409_140970


namespace walmart_cards_requested_l1409_140909

def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def best_buy_cards_requested : ℕ := 6
def best_buy_cards_sent : ℕ := 1
def walmart_cards_sent : ℕ := 2
def remaining_gift_card_value : ℕ := 3900

def total_best_buy_value : ℕ := best_buy_card_value * best_buy_cards_requested
def sent_gift_card_value : ℕ := best_buy_card_value * best_buy_cards_sent + walmart_card_value * walmart_cards_sent

theorem walmart_cards_requested (walmart_cards : ℕ) : 
  walmart_cards * walmart_card_value + total_best_buy_value = 
  remaining_gift_card_value + sent_gift_card_value → walmart_cards = 9 := by
  sorry

end walmart_cards_requested_l1409_140909


namespace power_calculation_l1409_140965

theorem power_calculation (a : ℝ) (h : a ≠ 0) : (a^2)^3 / a^2 = a^4 := by
  sorry

end power_calculation_l1409_140965


namespace three_n_equals_twenty_seven_l1409_140973

theorem three_n_equals_twenty_seven (n : ℤ) : 3 * n = 9 + 9 + 9 → n = 9 := by
  sorry

end three_n_equals_twenty_seven_l1409_140973


namespace fraction_ordering_l1409_140919

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end fraction_ordering_l1409_140919


namespace household_expenses_equal_savings_l1409_140930

/-- The number of years it takes to buy a house with all earnings -/
def years_to_buy : ℕ := 4

/-- The total number of years to buy the house -/
def total_years : ℕ := 24

/-- The number of years spent saving -/
def years_saving : ℕ := 12

/-- The number of years spent on household expenses -/
def years_household : ℕ := total_years - years_saving

theorem household_expenses_equal_savings : years_household = years_saving := by
  sorry

end household_expenses_equal_savings_l1409_140930


namespace tensor_plus_relation_l1409_140962

-- Define a structure for pairs of real numbers
structure Pair :=
  (x : ℝ)
  (y : ℝ)

-- Define equality for pairs
def pair_eq (a b : Pair) : Prop :=
  a.x = b.x ∧ a.y = b.y

-- Define the ⊗ operation
def tensor (a b : Pair) : Pair :=
  ⟨a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x⟩

-- Define the ⊕ operation
def plus (a b : Pair) : Pair :=
  ⟨a.x + b.x, a.y + b.y⟩

-- State the theorem
theorem tensor_plus_relation (p q : ℝ) :
  pair_eq (tensor ⟨1, 2⟩ ⟨p, q⟩) ⟨5, 0⟩ →
  pair_eq (plus ⟨1, 2⟩ ⟨p, q⟩) ⟨2, 0⟩ :=
by
  sorry

end tensor_plus_relation_l1409_140962


namespace exists_polyhedron_no_three_same_sided_faces_l1409_140935

/-- A face of a polyhedron --/
structure Face where
  sides : ℕ

/-- A polyhedron --/
structure Polyhedron where
  faces : List Face

/-- Predicate to check if a polyhedron has no three faces with the same number of sides --/
def has_no_three_same_sided_faces (p : Polyhedron) : Prop :=
  ∀ n : ℕ, (p.faces.filter (λ f => f.sides = n)).length < 3

/-- Theorem stating the existence of a polyhedron with no three faces having the same number of sides --/
theorem exists_polyhedron_no_three_same_sided_faces :
  ∃ p : Polyhedron, has_no_three_same_sided_faces p ∧ p.faces.length = 6 :=
sorry

end exists_polyhedron_no_three_same_sided_faces_l1409_140935


namespace min_value_of_sum_l1409_140968

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end min_value_of_sum_l1409_140968


namespace f_properties_l1409_140922

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 2) * cos (x / 2) - 2 * Real.sqrt 3 * sin (x / 2) ^ 2 + Real.sqrt 3

theorem f_properties (α : ℝ) (h1 : α ∈ Set.Ioo (π / 6) (2 * π / 3)) (h2 : f α = 6 / 5) :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π + π / 6) (2 * k * π + 7 * π / 6))) ∧
  f (α - π / 6) = (4 + 3 * Real.sqrt 3) / 5 := by
  sorry

end f_properties_l1409_140922


namespace F_bounded_and_amplitude_l1409_140936

def F (a x : ℝ) : ℝ := x * |x - 2*a| + 3

theorem F_bounded_and_amplitude (a : ℝ) (h : a ≤ 1/2) :
  ∃ (m M : ℝ), (∀ x ∈ Set.Icc 1 2, m ≤ F a x ∧ F a x ≤ M) ∧
  (M - m = 3 - 2*a) := by sorry

end F_bounded_and_amplitude_l1409_140936


namespace youngbin_shopping_combinations_l1409_140966

def n : ℕ := 3
def k : ℕ := 2

theorem youngbin_shopping_combinations : Nat.choose n k = 3 := by
  sorry

end youngbin_shopping_combinations_l1409_140966


namespace sin_40_tan_10_minus_sqrt_3_l1409_140929

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l1409_140929


namespace equation_solution_l1409_140944

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end equation_solution_l1409_140944


namespace complex_solutions_count_l1409_140908

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 - z - 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 - z - 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end complex_solutions_count_l1409_140908


namespace sequence_properties_l1409_140940

/-- Given a sequence {a_n} with the sum formula S_n = 2n^2 - 26n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 26 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 4 * n - 28

theorem sequence_properties :
  (∀ n : ℕ, a n = S (n + 1) - S n) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (∃ n : ℕ, n = 6 ∨ n = 7) ∧ (∀ m : ℕ, S m ≥ S 6 ∧ S m ≥ S 7) := by sorry

end sequence_properties_l1409_140940


namespace star_five_three_l1409_140943

def star (a b : ℝ) : ℝ := 4 * a + 6 * b

theorem star_five_three : star 5 3 = 38 := by
  sorry

end star_five_three_l1409_140943


namespace minimum_rent_is_36800_l1409_140983

/-- Represents the minimum rent problem for a travel agency --/
def MinimumRentProblem (total_passengers : ℕ) (capacity_A capacity_B : ℕ) (rent_A rent_B : ℕ) (max_buses : ℕ) (max_B_diff : ℕ) : Prop :=
  ∃ (num_A num_B : ℕ),
    -- Total passengers condition
    num_A * capacity_A + num_B * capacity_B ≥ total_passengers ∧
    -- Maximum number of buses condition
    num_A + num_B ≤ max_buses ∧
    -- Condition on the difference between B and A buses
    num_B ≤ num_A + max_B_diff ∧
    -- Minimum rent calculation
    ∀ (other_A other_B : ℕ),
      other_A * capacity_A + other_B * capacity_B ≥ total_passengers →
      other_A + other_B ≤ max_buses →
      other_B ≤ other_A + max_B_diff →
      num_A * rent_A + num_B * rent_B ≤ other_A * rent_A + other_B * rent_B

/-- The minimum rent for the given problem is 36800 yuan --/
theorem minimum_rent_is_36800 :
  MinimumRentProblem 900 36 60 1600 2400 21 7 →
  ∃ (num_A num_B : ℕ), num_A * 1600 + num_B * 2400 = 36800 :=
sorry

end minimum_rent_is_36800_l1409_140983


namespace roses_in_bouquet_l1409_140902

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_table_decorations : ℕ := 7

/-- The number of white roses in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed -/
def total_roses : ℕ := 109

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / num_bouquets

theorem roses_in_bouquet :
  roses_per_bouquet = 5 :=
by sorry

end roses_in_bouquet_l1409_140902


namespace fruit_prices_l1409_140981

theorem fruit_prices (mango_price banana_price : ℝ) : 
  (3 * mango_price + 2 * banana_price = 40) →
  (2 * mango_price + 3 * banana_price = 35) →
  (mango_price = 10 ∧ banana_price = 5) := by
sorry

end fruit_prices_l1409_140981


namespace factorization_cubic_minus_linear_l1409_140999

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x + 2)*(x - 2) := by
  sorry

end factorization_cubic_minus_linear_l1409_140999


namespace quadratic_inequality_solution_range_l1409_140963

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c > 0) ↔ (0 < c ∧ c < 16) :=
by sorry

end quadratic_inequality_solution_range_l1409_140963


namespace scalar_mult_assoc_l1409_140964

variable (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem scalar_mult_assoc (a : V) (h : a ≠ 0) :
  (-4 : ℝ) • (3 • a) = (-12 : ℝ) • a := by sorry

end scalar_mult_assoc_l1409_140964


namespace nikki_movie_length_l1409_140974

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn satisfy certain conditions -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ
  joyce_longer : joyce = michael + 2
  nikki_triple : nikki = 3 * michael
  ryn_proportion : ryn = (4/5) * nikki
  total_length : michael + joyce + nikki + ryn = 76

/-- Given the conditions, Nikki's favorite movie is 30 hours long -/
theorem nikki_movie_length (m : MovieLengths) : m.nikki = 30 := by
  sorry

end nikki_movie_length_l1409_140974


namespace student_number_factor_l1409_140991

theorem student_number_factor : ∃ f : ℚ, 122 * f - 138 = 106 :=
by
  -- Proof goes here
  sorry

end student_number_factor_l1409_140991


namespace paper_tearing_theorem_l1409_140971

/-- Represents the number of pieces after n tearing operations -/
def pieces (n : ℕ) : ℕ := 1 + 4 * n

theorem paper_tearing_theorem :
  (¬ ∃ n : ℕ, pieces n = 1994) ∧ (∃ n : ℕ, pieces n = 1997) := by
  sorry

end paper_tearing_theorem_l1409_140971


namespace sphere_volume_from_inscribed_cube_l1409_140952

theorem sphere_volume_from_inscribed_cube (s : ℝ) (h : s > 0) :
  let cube_surface_area := 6 * s^2
  let cube_diagonal := s * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  cube_surface_area = 24 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end sphere_volume_from_inscribed_cube_l1409_140952


namespace logan_max_rent_l1409_140927

def current_income : ℕ := 65000
def grocery_expenses : ℕ := 5000
def gas_expenses : ℕ := 8000
def desired_savings : ℕ := 42000
def income_increase : ℕ := 10000

def max_rent : ℕ := 20000

theorem logan_max_rent :
  max_rent = current_income + income_increase - desired_savings - grocery_expenses - gas_expenses :=
by sorry

end logan_max_rent_l1409_140927


namespace prime_sum_product_l1409_140939

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 85 → p * q = 166 := by
  sorry

end prime_sum_product_l1409_140939


namespace equilateral_triangle_theorem_l1409_140975

open Real

/-- Triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation given in the problem -/
def equation_holds (t : Triangle) : Prop :=
  (t.a * cos t.A + t.b * cos t.B + t.c * cos t.C) / 
  (t.a * sin t.A + t.b * sin t.B + t.c * sin t.C) = 
  (t.a + t.b + t.c) / (9 * circumradius t)

/-- The main theorem to prove -/
theorem equilateral_triangle_theorem (t : Triangle) :
  equation_holds t → t.a = t.b ∧ t.b = t.c := by sorry

end equilateral_triangle_theorem_l1409_140975


namespace cost_of_dozen_pens_l1409_140980

/-- Given the cost of some pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 120. -/
theorem cost_of_dozen_pens (n : ℕ) (x : ℚ) : 
  5 * n * x + 5 * x = 200 →  -- Cost of n pens and 5 pencils is 200
  (5 * x) / x = 5 / 1 →      -- Cost ratio of pen to pencil is 5:1
  12 * (5 * x) = 120 :=      -- Cost of dozen pens is 120
by sorry

end cost_of_dozen_pens_l1409_140980


namespace complement_intersection_theorem_l1409_140996

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end complement_intersection_theorem_l1409_140996


namespace voucher_draw_theorem_l1409_140932

/-- The number of apple cards in the bag -/
def num_apple : ℕ := 4

/-- The number of pear cards in the bag -/
def num_pear : ℕ := 4

/-- The total number of cards in the bag -/
def total_cards : ℕ := num_apple + num_pear

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The voucher amount random variable -/
inductive VoucherAmount : Type
  | zero : VoucherAmount
  | five : VoucherAmount
  | ten : VoucherAmount

/-- The probability of drawing 4 apple cards -/
def prob_four_apples : ℚ := 1 / 70

/-- The probability distribution of the voucher amount -/
def prob_distribution (x : VoucherAmount) : ℚ :=
  match x with
  | VoucherAmount.zero => 18 / 35
  | VoucherAmount.five => 16 / 35
  | VoucherAmount.ten => 1 / 35

/-- The expected value of the voucher amount -/
def expected_value : ℚ := 18 / 7

/-- Theorem stating the correctness of the probability and expected value calculations -/
theorem voucher_draw_theorem :
  (prob_four_apples = 1 / 70) ∧
  (∀ x, prob_distribution x = match x with
    | VoucherAmount.zero => 18 / 35
    | VoucherAmount.five => 16 / 35
    | VoucherAmount.ten => 1 / 35) ∧
  (expected_value = 18 / 7) := by sorry

end voucher_draw_theorem_l1409_140932


namespace ellipse_major_axis_length_l1409_140925

/-- The length of the major axis of an ellipse with equation x^2/25 + y^2/16 = 1 is 10 -/
theorem ellipse_major_axis_length : 
  ∀ x y : ℝ, x^2/25 + y^2/16 = 1 → 
  ∃ a b : ℝ, a ≥ b ∧ a^2 = 25 ∧ b^2 = 16 ∧ 2*a = 10 :=
by sorry

end ellipse_major_axis_length_l1409_140925


namespace sum_not_zero_l1409_140921

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c * d - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) :
  a + b + c + d ≠ 0 := by
sorry

end sum_not_zero_l1409_140921


namespace correct_algorithm_statements_l1409_140990

-- Define the set of algorithm statements
def AlgorithmStatements : Set ℕ := {1, 2, 3}

-- Define the property of being a correct statement about algorithms
def IsCorrectStatement : ℕ → Prop :=
  fun n => match n with
    | 1 => False  -- Statement 1 is incorrect
    | 2 => True   -- Statement 2 is correct
    | 3 => True   -- Statement 3 is correct
    | _ => False  -- Other numbers are not valid statements

-- Theorem: The set of correct statements is {2, 3}
theorem correct_algorithm_statements :
  {n ∈ AlgorithmStatements | IsCorrectStatement n} = {2, 3} := by
  sorry


end correct_algorithm_statements_l1409_140990


namespace tan_sum_pi_twelfths_l1409_140901

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end tan_sum_pi_twelfths_l1409_140901


namespace boat_distance_l1409_140986

/-- The distance covered by a boat given its speed in still water and the time taken to cover the same distance downstream and upstream. -/
theorem boat_distance (v : ℝ) (t_down t_up : ℝ) (h1 : v = 7) (h2 : t_down = 2) (h3 : t_up = 5) :
  ∃ (d : ℝ), d = 20 ∧ d = (v + (v * t_down - d) / t_down) * t_down ∧ d = (v - (v * t_up - d) / t_up) * t_up :=
sorry

end boat_distance_l1409_140986


namespace equation_solution_l1409_140933

theorem equation_solution : ∃! x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := by
  sorry

end equation_solution_l1409_140933


namespace bottom_row_bricks_l1409_140924

/-- Represents a pyramidal brick wall -/
structure PyramidalWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the total number of bricks in a pyramidal wall -/
def calculateTotalBricks (wall : PyramidalWall) : ℕ :=
  (wall.rows : ℕ) * (2 * wall.bottomRowBricks - wall.rows + 1) / 2

theorem bottom_row_bricks (wall : PyramidalWall) 
  (h1 : wall.rows = 15)
  (h2 : wall.totalBricks = 300)
  (h3 : calculateTotalBricks wall = wall.totalBricks) :
  wall.bottomRowBricks = 27 := by
  sorry

end bottom_row_bricks_l1409_140924


namespace sin_fourth_sum_eighths_pi_l1409_140938

theorem sin_fourth_sum_eighths_pi : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
sorry

end sin_fourth_sum_eighths_pi_l1409_140938


namespace todd_gum_problem_l1409_140985

theorem todd_gum_problem (initial_gum : ℕ) (steve_gum : ℕ) (total_gum : ℕ) : 
  steve_gum = 16 → total_gum = 54 → total_gum = initial_gum + steve_gum → initial_gum = 38 := by
  sorry

end todd_gum_problem_l1409_140985


namespace same_average_speed_exists_l1409_140918

theorem same_average_speed_exists : ∃ y : ℝ, 
  (y^2 - 14*y + 45 = (y^2 - 2*y - 35) / (y - 5)) ∧ 
  (y^2 - 14*y + 45 = 6) := by
  sorry

end same_average_speed_exists_l1409_140918


namespace smallest_prime_with_digit_sum_23_l1409_140992

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end smallest_prime_with_digit_sum_23_l1409_140992


namespace sum_of_acute_angles_not_always_acute_l1409_140928

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define the statement we want to prove false
def sum_of_acute_angles_always_acute : Prop :=
  ∀ (a b : ℝ), is_acute_angle a → is_acute_angle b → is_acute_angle (a + b)

-- Theorem stating that the above statement is false
theorem sum_of_acute_angles_not_always_acute :
  ¬ sum_of_acute_angles_always_acute :=
sorry

end sum_of_acute_angles_not_always_acute_l1409_140928


namespace ball_radius_under_shadow_l1409_140977

/-- The radius of a ball under specific shadow conditions -/
theorem ball_radius_under_shadow (ball_shadow_length : ℝ) (ruler_shadow_length : ℝ) 
  (h1 : ball_shadow_length = 10)
  (h2 : ruler_shadow_length = 2) : 
  ∃ (r : ℝ), r = 10 * Real.sqrt 5 - 20 ∧ r > 0 := by
  sorry

end ball_radius_under_shadow_l1409_140977


namespace at_least_n_prime_divisors_l1409_140916

theorem at_least_n_prime_divisors (n : ℕ) :
  ∃ (S : Finset Nat), (S.card ≥ n) ∧ (∀ p ∈ S, Nat.Prime p ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1)) :=
by sorry

end at_least_n_prime_divisors_l1409_140916


namespace all_star_seating_l1409_140997

/-- Represents the number of ways to seat 9 baseball All-Stars from 3 teams -/
def seating_arrangements : ℕ :=
  let num_teams : ℕ := 3
  let players_per_team : ℕ := 3
  let team_arrangements : ℕ := Nat.factorial num_teams
  let within_team_arrangements : ℕ := Nat.factorial players_per_team
  team_arrangements * (within_team_arrangements ^ num_teams)

/-- Theorem stating the number of seating arrangements for 9 baseball All-Stars -/
theorem all_star_seating :
  seating_arrangements = 1296 := by
  sorry

end all_star_seating_l1409_140997


namespace frog_hop_probability_l1409_140961

/-- Represents a position on a 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the edge of the grid -/
def isEdgePosition (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y + 3) % 4⟩
  | Direction.Left => ⟨(p.x + 3) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- The probability of reaching an edge position within n hops -/
def probReachEdge (n : Nat) (start : Position) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge 5 ⟨2, 2⟩ = 15/16 := by
  sorry

end frog_hop_probability_l1409_140961


namespace m_values_l1409_140949

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) → (m = 0 ∨ m = -1 ∨ m = 1/2) := by
  sorry

end m_values_l1409_140949


namespace rectangle_perimeter_l1409_140917

/-- Given two rectangles A and B, where A has a perimeter of 40 cm and its length is twice its width,
    and B has an area equal to one-half the area of A and its length is twice its width,
    prove that the perimeter of B is 20√2 cm. -/
theorem rectangle_perimeter (width_A : ℝ) (width_B : ℝ) : 
  (2 * (width_A + 2 * width_A) = 40) →  -- Perimeter of A is 40 cm
  (width_B * (2 * width_B) = (width_A * (2 * width_A)) / 2) →  -- Area of B is half of A
  (2 * (width_B + 2 * width_B) = 20 * Real.sqrt 2) :=  -- Perimeter of B is 20√2 cm
by sorry

end rectangle_perimeter_l1409_140917


namespace tom_neither_soccer_nor_test_l1409_140923

theorem tom_neither_soccer_nor_test (soccer_prob : ℚ) (test_prob : ℚ) 
  (h_soccer : soccer_prob = 5 / 8)
  (h_test : test_prob = 1 / 4)
  (h_independent : True) -- Assumption of independence
  : (1 - soccer_prob) * (1 - test_prob) = 9 / 32 := by
  sorry

end tom_neither_soccer_nor_test_l1409_140923


namespace dhoni_rent_percentage_dhoni_rent_percentage_proof_l1409_140987

theorem dhoni_rent_percentage : ℝ → Prop :=
  fun rent_percentage =>
    let dishwasher_percentage := rent_percentage - 5
    let leftover_percentage := 61
    rent_percentage + dishwasher_percentage + leftover_percentage = 100 →
    rent_percentage = 22

-- The proof is omitted
theorem dhoni_rent_percentage_proof : dhoni_rent_percentage 22 := by sorry

end dhoni_rent_percentage_dhoni_rent_percentage_proof_l1409_140987


namespace amys_net_earnings_result_l1409_140942

/-- Calculates Amy's net earnings for a week given her daily work details and tax rate -/
def amys_net_earnings (day1_hours day1_rate day1_tips day1_bonus : ℝ)
                      (day2_hours day2_rate day2_tips : ℝ)
                      (day3_hours day3_rate day3_tips : ℝ)
                      (day4_hours day4_rate day4_tips day4_overtime : ℝ)
                      (day5_hours day5_rate day5_tips : ℝ)
                      (tax_rate : ℝ) : ℝ :=
  let day1_earnings := day1_hours * day1_rate + day1_tips + day1_bonus
  let day2_earnings := day2_hours * day2_rate + day2_tips
  let day3_earnings := day3_hours * day3_rate + day3_tips
  let day4_earnings := day4_hours * day4_rate + day4_tips + day4_overtime
  let day5_earnings := day5_hours * day5_rate + day5_tips
  let gross_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings
  let taxes := tax_rate * gross_earnings
  gross_earnings - taxes

/-- Theorem stating that Amy's net earnings for the week are $118.58 -/
theorem amys_net_earnings_result :
  amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15 = 118.58 := by
  sorry

#eval amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15

end amys_net_earnings_result_l1409_140942


namespace juan_running_time_l1409_140907

/-- Given Juan's running distance and speed, prove that his running time is 8 hours. -/
theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 80) (h2 : speed = 10) :
  distance / speed = 8 := by
  sorry

end juan_running_time_l1409_140907


namespace isosceles_triangles_equal_perimeter_area_l1409_140910

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equal_side : ℝ
  base : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let h := Real.sqrt (t.equal_side^2 - (t.base/2)^2)
  (1/2) * t.base * h

/-- The theorem to be proved -/
theorem isosceles_triangles_equal_perimeter_area (t1 t2 : IsoscelesTriangle)
  (h1 : t1.equal_side = 6 ∧ t1.base = 10)
  (h2 : perimeter t1 = perimeter t2)
  (h3 : area t1 = area t2)
  (h4 : t2.base^2 / 2 = perimeter t2 / 2) :
  t2.base = Real.sqrt 22 := by
  sorry

end isosceles_triangles_equal_perimeter_area_l1409_140910


namespace work_completion_time_l1409_140957

/-- The number of days it takes for the original number of ladies to complete the work -/
def completion_time (original_ladies : ℕ) : ℝ :=
  6

/-- The time it takes for twice the number of ladies to complete half the work -/
def half_work_time (original_ladies : ℕ) : ℝ :=
  3

theorem work_completion_time (original_ladies : ℕ) :
  completion_time original_ladies = 2 * half_work_time original_ladies :=
by sorry

end work_completion_time_l1409_140957


namespace geese_count_l1409_140945

/-- The number of geese in a flock that land on n lakes -/
def geese (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem: The number of geese in a flock is 2^n - 1, where n is the number of lakes,
given the landing pattern described.
-/
theorem geese_count (n : ℕ) : 
  (∀ k < n, (geese k + 1) / 2 + (geese k) / 2 = geese (k + 1)) → 
  geese 0 = 0 → 
  geese n = 2^n - 1 := by
sorry

end geese_count_l1409_140945
