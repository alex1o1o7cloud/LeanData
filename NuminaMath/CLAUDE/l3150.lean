import Mathlib

namespace NUMINAMATH_CALUDE_salary_restoration_l3150_315099

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : 0 < original_salary) :
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := 1 + (1 / 3)
  reduced_salary * increase_factor = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l3150_315099


namespace NUMINAMATH_CALUDE_factorization_equality_l3150_315090

theorem factorization_equality (x : ℝ) : x * (x + 2) + (x + 2)^2 = 2 * (x + 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3150_315090


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3150_315061

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3150_315061


namespace NUMINAMATH_CALUDE_phrase_repetition_l3150_315023

/-- Represents a mapping from letters to words -/
def LetterWordMap (α : Type) := α → List α

/-- The result of applying the letter-to-word mapping n times -/
def applyMapping (n : ℕ) (map : LetterWordMap α) (initial : List α) : List α :=
  match n with
  | 0 => initial
  | n + 1 => applyMapping n map (initial.bind map)

theorem phrase_repetition 
  (α : Type) [Finite α] 
  (map : LetterWordMap α) 
  (initial : List α) 
  (h1 : initial.length ≥ 6) 
  (h2 : ∀ (a : α), (map a).length ≥ 1) :
  ∃ (i j : ℕ), 
    i ≠ j ∧ 
    i < (applyMapping 40 map initial).length ∧ 
    j < (applyMapping 40 map initial).length ∧
    (applyMapping 40 map initial).take 6 = 
      ((applyMapping 40 map initial).drop i).take 6 ∧
    (applyMapping 40 map initial).take 6 = 
      ((applyMapping 40 map initial).drop j).take 6 :=
by
  sorry


end NUMINAMATH_CALUDE_phrase_repetition_l3150_315023


namespace NUMINAMATH_CALUDE_solve_for_a_l3150_315070

theorem solve_for_a (x y a : ℚ) 
  (hx : x = 1)
  (hy : y = -2)
  (heq : 2 * x - a * y = 3) :
  a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3150_315070


namespace NUMINAMATH_CALUDE_inverse_mod_53_l3150_315088

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 31) : (36⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l3150_315088


namespace NUMINAMATH_CALUDE_billy_restaurant_bill_l3150_315004

/-- The total bill at Billy's Restaurant for a group with given characteristics -/
def total_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost : ℚ) 
  (num_fries_baskets : ℕ) (fries_basket_cost : ℚ) (drink_cost : ℚ) : ℚ :=
  num_adults * adult_meal_cost + 
  num_children * child_meal_cost + 
  num_fries_baskets * fries_basket_cost + 
  drink_cost

/-- Theorem stating that the total bill for the given group is $89 -/
theorem billy_restaurant_bill : 
  total_bill 4 3 12 7 2 5 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_billy_restaurant_bill_l3150_315004


namespace NUMINAMATH_CALUDE_subtracted_value_l3150_315037

theorem subtracted_value (N V : ℝ) (h1 : N = 1152) (h2 : N / 6 - V = 3) : V = 189 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3150_315037


namespace NUMINAMATH_CALUDE_trains_at_initial_positions_l3150_315077

/-- Represents a metro line with a given cycle time -/
structure MetroLine where
  cycletime : ℕ

/-- Represents a metro system with multiple lines -/
structure MetroSystem where
  lines : List MetroLine

/-- Checks if all trains return to their initial positions after a given time -/
def allTrainsAtInitialPositions (system : MetroSystem) (time : ℕ) : Prop :=
  ∀ line ∈ system.lines, time % line.cycletime = 0

/-- The metro system of city N -/
def cityNMetro : MetroSystem :=
  { lines := [
      { cycletime := 14 },  -- Red line
      { cycletime := 16 },  -- Blue line
      { cycletime := 18 }   -- Green line
    ]
  }

/-- Theorem: After 2016 minutes, all trains in city N's metro system will be at their initial positions -/
theorem trains_at_initial_positions :
  allTrainsAtInitialPositions cityNMetro 2016 :=
by
  sorry


end NUMINAMATH_CALUDE_trains_at_initial_positions_l3150_315077


namespace NUMINAMATH_CALUDE_marbles_fraction_taken_l3150_315018

theorem marbles_fraction_taken (chris_marbles ryan_marbles remaining_marbles : ℕ) 
  (h1 : chris_marbles = 12)
  (h2 : ryan_marbles = 28)
  (h3 : remaining_marbles = 20) :
  (chris_marbles + ryan_marbles - remaining_marbles) / (chris_marbles + ryan_marbles) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_fraction_taken_l3150_315018


namespace NUMINAMATH_CALUDE_correct_sets_count_l3150_315049

/-- A set of weights is represented as a multiset of natural numbers -/
def WeightSet := Multiset ℕ

/-- A weight set is correct if it satisfies the given conditions -/
def is_correct_set (s : WeightSet) : Prop :=
  (s.sum = 200) ∧
  (∀ w : ℕ, w ≤ 200 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of correct weight sets -/
def num_correct_sets : ℕ := 3

theorem correct_sets_count :
  ∃ (sets : Finset WeightSet),
    sets.card = num_correct_sets ∧
    (∀ s : WeightSet, s ∈ sets ↔ is_correct_set s) :=
sorry

end NUMINAMATH_CALUDE_correct_sets_count_l3150_315049


namespace NUMINAMATH_CALUDE_final_mango_distribution_l3150_315031

/-- Represents the state of mango distribution among friends in a circle. -/
structure MangoDistribution :=
  (friends : ℕ)
  (mangos : List ℕ)

/-- Defines the rules for sharing mangos. -/
def share (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Defines the rules for eating mangos. -/
def eat (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Checks if any further actions (sharing or eating) are possible. -/
def canContinue (d : MangoDistribution) : Bool :=
  sorry

/-- Applies sharing and eating rules until no further actions are possible. -/
def applyRulesUntilStable (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Counts the number of people with mangos in the final distribution. -/
def countPeopleWithMangos (d : MangoDistribution) : ℕ :=
  sorry

/-- Main theorem stating that exactly 8 people will have mangos at the end. -/
theorem final_mango_distribution
  (initial : MangoDistribution)
  (h1 : initial.friends = 100)
  (h2 : initial.mangos = [2019] ++ List.replicate 99 0) :
  countPeopleWithMangos (applyRulesUntilStable initial) = 8 :=
sorry

end NUMINAMATH_CALUDE_final_mango_distribution_l3150_315031


namespace NUMINAMATH_CALUDE_sick_children_count_l3150_315086

/-- Calculates the number of children who called in sick given the initial number of jellybeans,
    normal class size, jellybeans eaten per child, and jellybeans left. -/
def children_called_sick (initial_jellybeans : ℕ) (normal_class_size : ℕ) 
                         (jellybeans_per_child : ℕ) (jellybeans_left : ℕ) : ℕ :=
  normal_class_size - (initial_jellybeans - jellybeans_left) / jellybeans_per_child

theorem sick_children_count : 
  children_called_sick 100 24 3 34 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sick_children_count_l3150_315086


namespace NUMINAMATH_CALUDE_brother_papaya_consumption_l3150_315036

/-- The number of papayas Jake eats in one week -/
def jake_weekly : ℕ := 3

/-- The number of papayas Jake's father eats in one week -/
def father_weekly : ℕ := 4

/-- The total number of papayas needed for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of papayas Jake's brother eats in one week -/
def brother_weekly : ℕ := 5

theorem brother_papaya_consumption :
  4 * (jake_weekly + father_weekly + brother_weekly) = total_papayas := by
  sorry

end NUMINAMATH_CALUDE_brother_papaya_consumption_l3150_315036


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l3150_315060

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℕ := 25

/-- Represents the full year salary in Rupees -/
def full_year_salary : ℕ := 900

/-- Represents the number of months served -/
def months_served : ℕ := 9

/-- Represents the actual payment received for the partial service in Rupees -/
def partial_payment : ℕ := 650

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

theorem uniform_price_calculation :
  uniform_price = (full_year_salary * months_served / months_in_year) - partial_payment := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l3150_315060


namespace NUMINAMATH_CALUDE_sibling_ages_problem_l3150_315025

/-- The current age of the eldest sibling given the conditions of the problem -/
def eldest_age : ℕ := 20

theorem sibling_ages_problem :
  let second_age := eldest_age - 5
  let youngest_age := second_age - 5
  let future_sum := (eldest_age + 10) + (second_age + 10) + (youngest_age + 10)
  future_sum = 75 ∧ eldest_age = 20 := by sorry

end NUMINAMATH_CALUDE_sibling_ages_problem_l3150_315025


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3150_315006

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -4 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (2 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 2, -4 - (-6))
  let projection := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let perpendicular := (v.1 - projection.1, v.2 - projection.2)
  Real.sqrt (perpendicular.1^2 + perpendicular.2^2) = Real.sqrt 2349 / 29 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3150_315006


namespace NUMINAMATH_CALUDE_star_example_l3150_315019

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (10/6) = 75 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l3150_315019


namespace NUMINAMATH_CALUDE_parakeet_consumption_l3150_315097

/-- Represents the daily birdseed consumption of each bird type and the total for a week -/
structure BirdseedConsumption where
  parakeet : ℝ
  parrot : ℝ
  finch : ℝ
  total_weekly : ℝ

/-- The number of each type of bird -/
structure BirdCounts where
  parakeets : ℕ
  parrots : ℕ
  finches : ℕ

/-- Calculates the total daily consumption for all birds -/
def total_daily_consumption (c : BirdseedConsumption) (b : BirdCounts) : ℝ :=
  c.parakeet * b.parakeets + c.parrot * b.parrots + c.finch * b.finches

/-- Theorem stating the daily consumption of each parakeet -/
theorem parakeet_consumption (c : BirdseedConsumption) (b : BirdCounts) :
  c.parakeet = 2 ∧
  c.parrot = 14 ∧
  c.finch = c.parakeet / 2 ∧
  b.parakeets = 3 ∧
  b.parrots = 2 ∧
  b.finches = 4 ∧
  c.total_weekly = 266 ∧
  c.total_weekly = 7 * total_daily_consumption c b :=
by sorry

end NUMINAMATH_CALUDE_parakeet_consumption_l3150_315097


namespace NUMINAMATH_CALUDE_nicole_collected_400_cards_l3150_315033

/-- The number of Pokemon cards Nicole collected -/
def nicole_cards : ℕ := 400

/-- The number of Pokemon cards Cindy collected -/
def cindy_cards : ℕ := 2 * nicole_cards

/-- The number of Pokemon cards Rex collected -/
def rex_cards : ℕ := (nicole_cards + cindy_cards) / 2

/-- The number of people Rex divided his cards among (himself and 3 siblings) -/
def num_people : ℕ := 4

/-- The number of cards Rex has left after dividing -/
def rex_leftover : ℕ := 150

theorem nicole_collected_400_cards :
  nicole_cards = 400 ∧
  cindy_cards = 2 * nicole_cards ∧
  rex_cards = (nicole_cards + cindy_cards) / 2 ∧
  rex_cards = num_people * rex_leftover :=
sorry

end NUMINAMATH_CALUDE_nicole_collected_400_cards_l3150_315033


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3150_315007

theorem log_sum_equals_two : Real.log 4 + 2 * Real.log 5 = 2 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3150_315007


namespace NUMINAMATH_CALUDE_complex_power_36_l3150_315048

theorem complex_power_36 :
  (Complex.exp (160 * π / 180 * Complex.I))^36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_power_36_l3150_315048


namespace NUMINAMATH_CALUDE_frog_eggs_first_day_l3150_315074

/-- Represents the number of eggs laid by a frog over 4 days -/
def frog_eggs (x : ℕ) : ℕ :=
  let day1 := x
  let day2 := 2 * x
  let day3 := 2 * x + 20
  let day4 := 2 * (day1 + day2 + day3)
  day1 + day2 + day3 + day4

/-- Theorem stating that if the frog lays 810 eggs over 4 days following the given pattern,
    then it laid 50 eggs on the first day -/
theorem frog_eggs_first_day :
  ∃ (x : ℕ), frog_eggs x = 810 ∧ x = 50 :=
sorry

end NUMINAMATH_CALUDE_frog_eggs_first_day_l3150_315074


namespace NUMINAMATH_CALUDE_max_d_value_l3150_315079

def a (n : ℕ) : ℕ := 99 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (M : ℕ), M = 401 ∧ ∀ (n : ℕ), n > 0 → d n ≤ M ∧ ∃ (k : ℕ), k > 0 ∧ d k = M :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3150_315079


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3150_315096

theorem sin_alpha_value (α : Real) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3150_315096


namespace NUMINAMATH_CALUDE_photos_per_album_l3150_315082

theorem photos_per_album 
  (total_photos : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_photos = 2560) 
  (h2 : num_albums = 32) 
  (h3 : total_photos % num_albums = 0) :
  total_photos / num_albums = 80 := by
sorry

end NUMINAMATH_CALUDE_photos_per_album_l3150_315082


namespace NUMINAMATH_CALUDE_exponential_sum_conjugate_l3150_315016

theorem exponential_sum_conjugate (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 5/8 * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = -1/3 - 5/8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_exponential_sum_conjugate_l3150_315016


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l3150_315045

theorem opposite_numbers_equation (x : ℚ) : 
  x / 5 + (3 - 2 * x) / 2 = 0 → x = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l3150_315045


namespace NUMINAMATH_CALUDE_olivias_initial_amount_l3150_315089

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 25

/-- The amount of money Olivia had left after visiting the supermarket -/
def amount_left : ℕ := 29

/-- Theorem stating that Olivia's initial amount of money was $54 -/
theorem olivias_initial_amount : initial_amount = 54 := by sorry

end NUMINAMATH_CALUDE_olivias_initial_amount_l3150_315089


namespace NUMINAMATH_CALUDE_dinner_price_problem_l3150_315027

theorem dinner_price_problem (original_price : ℝ) : 
  -- John's payment (after discount and tip)
  (0.90 * original_price + 0.15 * original_price) -
  -- Jane's payment (after discount and tip)
  (0.90 * original_price + 0.15 * (0.90 * original_price)) = 0.54 →
  original_price = 36 := by
sorry

end NUMINAMATH_CALUDE_dinner_price_problem_l3150_315027


namespace NUMINAMATH_CALUDE_solution_pairs_l3150_315028

theorem solution_pairs : 
  ∀ x y : ℝ, 
    (x^2 + y^2 + x + y = x*y*(x + y) - 10/27 ∧ 
     |x*y| ≤ 25/9) ↔ 
    ((x = -1/3 ∧ y = -1/3) ∨ (x = 5/3 ∧ y = 5/3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l3150_315028


namespace NUMINAMATH_CALUDE_range_of_f_l3150_315022

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3150_315022


namespace NUMINAMATH_CALUDE_second_child_birth_year_l3150_315056

/-- 
Given a couple married in 1980 with two children, one born in 1982 and the other
in an unknown year, if their combined ages equal the years of marriage in 1986,
then the second child was born in 1992.
-/
theorem second_child_birth_year 
  (marriage_year : Nat) 
  (first_child_birth_year : Nat) 
  (second_child_birth_year : Nat) 
  (h1 : marriage_year = 1980)
  (h2 : first_child_birth_year = 1982)
  (h3 : (1986 - first_child_birth_year) + (1986 - second_child_birth_year) + (1986 - marriage_year) = 1986) :
  second_child_birth_year = 1992 := by
sorry

end NUMINAMATH_CALUDE_second_child_birth_year_l3150_315056


namespace NUMINAMATH_CALUDE_function_inequality_l3150_315078

/-- A function satisfying the given conditions -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, (x * (deriv f x) - f x) ≤ 0

theorem function_inequality
  (f : ℝ → ℝ)
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : Differentiable ℝ f)
  (h_cond : SatisfiesCondition f)
  (m n : ℝ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_lt : m < n) :
  m * f n ≤ n * f m :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l3150_315078


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3150_315010

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : monotone_decreasing_on f (Set.Ici 0))
  (h_f1 : f 1 = 0) :
  {x | f x > 0} = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3150_315010


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3150_315080

/-- A parallelogram with vertices A, B, C, and D in a real inner product space. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_parallelogram : A - B = D - C)

/-- The theorem stating that if BD = 2 and 2(AD • AB) = |BC|^2 in a parallelogram ABCD,
    then |AB| = 2. -/
theorem parallelogram_side_length
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (para : Parallelogram V)
  (h1 : ‖para.B - para.D‖ = 2)
  (h2 : 2 * inner (para.A - para.D) (para.A - para.B) = ‖para.B - para.C‖^2) :
  ‖para.A - para.B‖ = 2 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3150_315080


namespace NUMINAMATH_CALUDE_sine_cosine_zero_points_l3150_315064

theorem sine_cosine_zero_points (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)
  (∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, 0 < x ∧ x < 4 * Real.pi ∧ f x = 0) →
  7 / 6 < ω ∧ ω ≤ 17 / 12 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_zero_points_l3150_315064


namespace NUMINAMATH_CALUDE_fruit_basket_difference_l3150_315094

/-- Proof that the difference between oranges and apples is 2 in a fruit basket -/
theorem fruit_basket_difference : ∀ (apples bananas peaches : ℕ),
  apples + bananas + peaches + 6 = 28 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  6 - apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_difference_l3150_315094


namespace NUMINAMATH_CALUDE_quadratic_completion_l3150_315032

theorem quadratic_completion (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + (1/5 : ℝ) = (x + n)^2 + (1/20 : ℝ)) → b < 0 → b = -Real.sqrt (3/5)
:= by sorry

end NUMINAMATH_CALUDE_quadratic_completion_l3150_315032


namespace NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l3150_315059

theorem triangle_angle_sixty_degrees (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  C = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l3150_315059


namespace NUMINAMATH_CALUDE_fangfang_floor_climb_l3150_315095

def time_between_floors (start_floor end_floor : ℕ) (time : ℝ) : Prop :=
  time = (end_floor - start_floor) * 15

theorem fangfang_floor_climb : 
  time_between_floors 1 3 30 → time_between_floors 2 6 60 :=
by
  sorry

end NUMINAMATH_CALUDE_fangfang_floor_climb_l3150_315095


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l3150_315046

/-- Calculates the total fuel consumption over six weeks given specific conditions for each week. -/
theorem total_fuel_consumption
  (week1_consumption : ℝ)
  (week2_increase : ℝ)
  (week3_fraction : ℝ)
  (week4_increase : ℝ)
  (week5_budget : ℝ)
  (week5_price : ℝ)
  (week6_increase : ℝ)
  (h1 : week1_consumption = 25)
  (h2 : week2_increase = 0.1)
  (h3 : week3_fraction = 0.5)
  (h4 : week4_increase = 0.3)
  (h5 : week5_budget = 50)
  (h6 : week5_price = 2.5)
  (h7 : week6_increase = 0.2) :
  week1_consumption +
  (week1_consumption * (1 + week2_increase)) +
  (week1_consumption * week3_fraction) +
  (week1_consumption * week3_fraction * (1 + week4_increase)) +
  (week5_budget / week5_price) +
  (week5_budget / week5_price * (1 + week6_increase)) = 125.25 :=
by sorry

end NUMINAMATH_CALUDE_total_fuel_consumption_l3150_315046


namespace NUMINAMATH_CALUDE_smallest_first_term_arithmetic_progression_l3150_315068

theorem smallest_first_term_arithmetic_progression 
  (S₃ S₆ : ℕ) (d₁ : ℚ) 
  (h₁ : d₁ ≥ 1/2) 
  (h₂ : S₃ = 3 * d₁ + 3 * (S₆ - 2 * S₃) / 3) 
  (h₃ : S₆ = 6 * d₁ + 15 * (S₆ - 2 * S₃) / 3) :
  d₁ ≥ 5/9 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_arithmetic_progression_l3150_315068


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l3150_315076

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The interval (-∞, 4] -/
def interval : Set ℝ := Set.Iic 4

theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ∈ interval, StrictMonoOn (f a) interval) → a < -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l3150_315076


namespace NUMINAMATH_CALUDE_cube_difference_equals_36_l3150_315087

theorem cube_difference_equals_36 (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_equals_36_l3150_315087


namespace NUMINAMATH_CALUDE_unique_p_type_prime_l3150_315073

/-- A prime number q is a P-type prime if q + 1 is a perfect square. -/
def is_p_type_prime (q : ℕ) : Prop :=
  Nat.Prime q ∧ ∃ m : ℕ, q + 1 = m^2

/-- There exists exactly one P-type prime number. -/
theorem unique_p_type_prime : ∃! q : ℕ, is_p_type_prime q :=
sorry

end NUMINAMATH_CALUDE_unique_p_type_prime_l3150_315073


namespace NUMINAMATH_CALUDE_average_monthly_salary_l3150_315081

/-- Calculates the average monthly salary of five employees given their base salaries and bonus/deduction percentages. -/
theorem average_monthly_salary
  (base_A base_B base_C base_D base_E : ℕ)
  (bonus_A bonus_B1 bonus_D bonus_E : ℚ)
  (deduct_B deduct_D deduct_E : ℚ)
  (h_base_A : base_A = 8000)
  (h_base_B : base_B = 5000)
  (h_base_C : base_C = 16000)
  (h_base_D : base_D = 7000)
  (h_base_E : base_E = 9000)
  (h_bonus_A : bonus_A = 5 / 100)
  (h_bonus_B1 : bonus_B1 = 10 / 100)
  (h_deduct_B : deduct_B = 2 / 100)
  (h_bonus_D : bonus_D = 8 / 100)
  (h_deduct_D : deduct_D = 3 / 100)
  (h_bonus_E : bonus_E = 12 / 100)
  (h_deduct_E : deduct_E = 5 / 100) :
  (base_A * (1 + bonus_A) +
   base_B * (1 + bonus_B1 - deduct_B) +
   base_C +
   base_D * (1 + bonus_D - deduct_D) +
   base_E * (1 + bonus_E - deduct_E)) / 5 = 8756 :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_salary_l3150_315081


namespace NUMINAMATH_CALUDE_power_mod_nineteen_l3150_315050

theorem power_mod_nineteen : 2^65537 % 19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_nineteen_l3150_315050


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l3150_315003

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  height : ℝ
  circum_radius : ℝ
  height_ratio : height / circum_radius = Real.sqrt (2/3)

/-- The angles of an isosceles trapezoid -/
def trapezoid_angles (t : IsoscelesTrapezoid) : ℝ × ℝ := sorry

theorem isosceles_trapezoid_angles (t : IsoscelesTrapezoid) :
  trapezoid_angles t = (45, 135) := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l3150_315003


namespace NUMINAMATH_CALUDE_problem_statement_l3150_315002

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
  (¬(∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x - Real.log x ≤ 0)) ∧
  (∀ p q : Prop, (p ∨ q → p ∧ q) → False) ∧
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3150_315002


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3150_315055

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ b < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3150_315055


namespace NUMINAMATH_CALUDE_jewelry_ensemble_orders_l3150_315014

theorem jewelry_ensemble_orders (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earrings_sold : ℕ)
  (total_amount : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : necklaces_sold = 5)
  (h6 : bracelets_sold = 10)
  (h7 : earrings_sold = 20)
  (h8 : total_amount = 565) :
  (total_amount - (necklace_price * necklaces_sold + bracelet_price * bracelets_sold + earring_price * earrings_sold)) / ensemble_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_ensemble_orders_l3150_315014


namespace NUMINAMATH_CALUDE_total_candies_l3150_315015

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's candies -/
def caleb_jellybeans : ℕ := 3 * dozen
def caleb_chocolate_bars : ℕ := 5
def caleb_gummy_bears : ℕ := 8

/-- Sophie's candies -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2
def sophie_chocolate_bars : ℕ := 3
def sophie_gummy_bears : ℕ := 12

/-- Max's candies -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen
def max_chocolate_bars : ℕ := 6
def max_gummy_bears : ℕ := 10

/-- Total candies for each person -/
def caleb_total : ℕ := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears
def sophie_total : ℕ := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears
def max_total : ℕ := max_jellybeans + max_chocolate_bars + max_gummy_bears

/-- Theorem: The total number of candies is 140 -/
theorem total_candies : caleb_total + sophie_total + max_total = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3150_315015


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l3150_315084

theorem min_value_2a_plus_b (a b : ℝ) (h : Real.log a + Real.log b = Real.log (a + 2*b)) :
  (∀ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) → 2*x + y ≥ 2*a + b) ∧ (∃ x y : ℝ, Real.log x + Real.log y = Real.log (x + 2*y) ∧ 2*x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l3150_315084


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3150_315021

theorem complex_equation_solution (x y : ℝ) :
  Complex.I * (x + y) = x - 1 → x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3150_315021


namespace NUMINAMATH_CALUDE_visitors_to_both_countries_l3150_315030

theorem visitors_to_both_countries (total : ℕ) (iceland : ℕ) (norway : ℕ) (neither : ℕ) : 
  total = 90 → iceland = 55 → norway = 33 → neither = 53 → 
  total - neither = iceland + norway - (total - neither) := by
  sorry

end NUMINAMATH_CALUDE_visitors_to_both_countries_l3150_315030


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3150_315054

def number_of_maple_trees : ℕ := 3
def number_of_oak_trees : ℕ := 4
def number_of_birch_trees : ℕ := 5
def total_trees : ℕ := number_of_maple_trees + number_of_oak_trees + number_of_birch_trees

def probability_no_adjacent_birch : ℚ := 7 / 99

theorem birch_tree_arrangement_probability :
  probability_no_adjacent_birch = 
    (Nat.choose (number_of_maple_trees + number_of_oak_trees + 1) number_of_birch_trees) / 
    (Nat.choose total_trees number_of_birch_trees) :=
sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3150_315054


namespace NUMINAMATH_CALUDE_y_axis_intersection_l3150_315009

/-- The line equation 4y + 3x = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * y + 3 * x = 24

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_axis_intersection_l3150_315009


namespace NUMINAMATH_CALUDE_max_books_robert_can_buy_l3150_315042

theorem max_books_robert_can_buy (book_cost : ℚ) (available_money : ℚ) : 
  book_cost = 875/100 → available_money = 250 → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_books_robert_can_buy_l3150_315042


namespace NUMINAMATH_CALUDE_percentage_of_200_rupees_l3150_315043

theorem percentage_of_200_rupees : (25 / 100 : ℚ) * 200 = 50 := by sorry

end NUMINAMATH_CALUDE_percentage_of_200_rupees_l3150_315043


namespace NUMINAMATH_CALUDE_least_distinct_values_in_list_l3150_315098

theorem least_distinct_values_in_list (list : List ℕ) : 
  list.length = 2030 →
  ∃! m, m ∈ list ∧ (list.count m = 11) ∧ (∀ n ∈ list, n ≠ m → list.count n < 11) →
  (∃ x : ℕ, x = list.toFinset.card ∧ x ≥ 203 ∧ ∀ y : ℕ, y = list.toFinset.card → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_in_list_l3150_315098


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3150_315017

theorem derivative_x_squared_cos_x (x : ℝ) :
  deriv (fun x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3150_315017


namespace NUMINAMATH_CALUDE_a_range_l3150_315085

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3*a else a^x - 2

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (a > 0 ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_a_range_l3150_315085


namespace NUMINAMATH_CALUDE_log_eight_x_three_halves_l3150_315005

theorem log_eight_x_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_three_halves_l3150_315005


namespace NUMINAMATH_CALUDE_existence_of_special_function_l3150_315067

theorem existence_of_special_function :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 1993 * n^1945 :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_function_l3150_315067


namespace NUMINAMATH_CALUDE_a_in_M_neither_sufficient_nor_necessary_for_a_in_N_l3150_315053

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is neither sufficient nor necessary for "a ∈ N"
theorem a_in_M_neither_sufficient_nor_necessary_for_a_in_N :
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) ∧ (∃ a : ℝ, a ∈ N ∧ a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_neither_sufficient_nor_necessary_for_a_in_N_l3150_315053


namespace NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3150_315039

/-- Given three quadratic polynomials and a condition on their coefficients,
    prove that at least one polynomial has a positive discriminant. -/
theorem at_least_one_positive_discriminant
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (h' : b₁ * b₂ * b₃ > 1) :
  (4 * a₁^2 - 4 * b₁ > 0) ∨ (4 * a₂^2 - 4 * b₂ > 0) ∨ (4 * a₃^2 - 4 * b₃ > 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3150_315039


namespace NUMINAMATH_CALUDE_x_value_l3150_315057

theorem x_value (x : ℚ) 
  (eq1 : 9 * x^2 + 8 * x - 1 = 0) 
  (eq2 : 27 * x^2 + 65 * x - 8 = 0) : 
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_x_value_l3150_315057


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_20_l3150_315012

theorem smallest_n_divisible_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧ n = 9 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 →
    ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b c d : ℤ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_20_l3150_315012


namespace NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l3150_315001

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y : ℝ, f x = y ↔ f (-x) = y :=
by sorry

end NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l3150_315001


namespace NUMINAMATH_CALUDE_sector_arc_length_l3150_315020

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (L : ℝ) : 
  r = 1 → θ_deg = 60 → L = r * (θ_deg * π / 180) → L = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3150_315020


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3150_315062

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3150_315062


namespace NUMINAMATH_CALUDE_remainder_sum_mod_five_l3150_315093

theorem remainder_sum_mod_five : (9^5 + 8^7 + 7^6) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_five_l3150_315093


namespace NUMINAMATH_CALUDE_fraction_inequality_l3150_315038

theorem fraction_inequality (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3150_315038


namespace NUMINAMATH_CALUDE_cakes_served_today_l3150_315071

theorem cakes_served_today (dinner_stock : ℕ) (lunch_served : ℚ) (dinner_percentage : ℚ) :
  dinner_stock = 95 →
  lunch_served = 48.5 →
  dinner_percentage = 62.25 →
  ⌈lunch_served + (dinner_percentage / 100) * dinner_stock⌉ = 108 :=
by sorry

end NUMINAMATH_CALUDE_cakes_served_today_l3150_315071


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_25_l3150_315063

/-- Given a man's speed against a current and the current's speed, 
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 25 km/hr. -/
theorem mans_speed_with_current_is_25 :
  mans_speed_with_current 20 2.5 = 25 := by
  sorry

#eval mans_speed_with_current 20 2.5

end NUMINAMATH_CALUDE_mans_speed_with_current_is_25_l3150_315063


namespace NUMINAMATH_CALUDE_cylinder_volume_l3150_315000

/-- Given a cylinder and a cone with specific ratios of heights and base circumferences,
    and a known volume of the cone, prove the volume of the cylinder. -/
theorem cylinder_volume (h_cyl h_cone r_cyl r_cone : ℝ) (vol_cone : ℝ) : 
  h_cyl / h_cone = 4 / 5 →
  r_cyl / r_cone = 3 / 5 →
  vol_cone = 250 →
  (π * r_cyl^2 * h_cyl : ℝ) = 216 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3150_315000


namespace NUMINAMATH_CALUDE_twelve_valid_grids_l3150_315047

/-- Represents a 3x3 grid filled with numbers 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if the grid is valid according to the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  g 1 1 = 0 ∧  -- 1 is in top-left corner
  g 3 3 = 8 ∧  -- 9 is in bottom-right corner
  g 2 2 = 3 ∧  -- 4 is in center
  (∀ i j, i < j → g i 1 < g j 1) ∧  -- increasing top to bottom in first column
  (∀ i j, i < j → g i 2 < g j 2) ∧  -- increasing top to bottom in second column
  (∀ i j, i < j → g i 3 < g j 3) ∧  -- increasing top to bottom in third column
  (∀ i j, i < j → g 1 i < g 1 j) ∧  -- increasing left to right in first row
  (∀ i j, i < j → g 2 i < g 2 j) ∧  -- increasing left to right in second row
  (∀ i j, i < j → g 3 i < g 3 j)    -- increasing left to right in third row

/-- The number of valid grid arrangements -/
def num_valid_grids : ℕ := sorry

/-- Theorem stating that there are exactly 12 valid grid arrangements -/
theorem twelve_valid_grids : num_valid_grids = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_valid_grids_l3150_315047


namespace NUMINAMATH_CALUDE_oscar_swag_bag_scarves_l3150_315091

/-- Represents the contents and value of an Oscar swag bag -/
structure SwagBag where
  totalValue : ℕ
  earringCost : ℕ
  iphoneCost : ℕ
  scarfCost : ℕ
  numScarves : ℕ

/-- Theorem stating that given the specific costs and total value, 
    the number of scarves in the swag bag is 4 -/
theorem oscar_swag_bag_scarves (bag : SwagBag) 
    (h1 : bag.totalValue = 20000)
    (h2 : bag.earringCost = 6000)
    (h3 : bag.iphoneCost = 2000)
    (h4 : bag.scarfCost = 1500)
    (h5 : bag.totalValue = 2 * bag.earringCost + bag.iphoneCost + bag.numScarves * bag.scarfCost) :
  bag.numScarves = 4 := by
  sorry

#check oscar_swag_bag_scarves

end NUMINAMATH_CALUDE_oscar_swag_bag_scarves_l3150_315091


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l3150_315013

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z / (1 + 2 * I) = 1 - 2 * I) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l3150_315013


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_third_term_l3150_315083

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n : ℕ, a n = first_term * q^n + common_diff * (1 - q^n) / (1 - q)

/-- Sum of first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithmeticGeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * (1 - seq.q^n) / (1 - seq.q)

theorem arithmetic_geometric_sequence_third_term
  (seq : ArithmeticGeometricSequence)
  (h1 : sum_n seq 6 / sum_n seq 3 = -19/8)
  (h2 : seq.a 4 - seq.a 2 = -15/8) :
  seq.a 3 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_third_term_l3150_315083


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3150_315026

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The problem statement -/
theorem product_of_binary_and_ternary : 
  let binary := [1, 0, 1, 1]  -- 1101 in binary, least significant digit first
  let ternary := [2, 0, 2]    -- 202 in ternary, least significant digit first
  (to_decimal binary 2) * (to_decimal ternary 3) = 260 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3150_315026


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3150_315075

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ 
  a 2 * a 3 = 2 * a 1 ∧ 
  (a 4 + 2 * a 7) / 2 = 5 / 4

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  SequenceConditions a → a 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3150_315075


namespace NUMINAMATH_CALUDE_negative_root_condition_l3150_315066

theorem negative_root_condition (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x+1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_root_condition_l3150_315066


namespace NUMINAMATH_CALUDE_simplify_fraction_l3150_315069

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -3) :
  (x + 4) / (x^2 + 3*x) - 1 / (3*x + x^2) = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3150_315069


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3150_315065

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = vertex_sum := by
sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3150_315065


namespace NUMINAMATH_CALUDE_propositions_true_l3150_315058

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of real roots for a quadratic equation
def has_real_roots (b : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0

theorem propositions_true : 
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧
  (∀ b : ℝ, ¬(has_real_roots b) → b > -1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l3150_315058


namespace NUMINAMATH_CALUDE_sum_m_n_equals_three_l3150_315035

theorem sum_m_n_equals_three (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + 5 < n)
  (h4 : (m + (m + 3) + (m + 5) + n + (n + 2) + (2 * n - 1)) / 6 = n + 1)
  (h5 : ((m + 5) + n) / 2 = n + 1) : m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_three_l3150_315035


namespace NUMINAMATH_CALUDE_sequence_problem_solution_l3150_315011

def sequence_problem (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n * a (n + 1) = 1 - a (n + 1)) ∧ 
  (a 2010 = 2) ∧
  (a 2008 = -3)

theorem sequence_problem_solution :
  ∃ a : ℕ → ℝ, sequence_problem a :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_solution_l3150_315011


namespace NUMINAMATH_CALUDE_hcl_production_l3150_315072

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℕ  -- moles of NaCl
  reactant2 : ℕ  -- moles of HNO3
  product : ℕ    -- moles of HCl produced

-- Define the stoichiometric relationship
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.product = min r.reactant1 r.reactant2

-- Theorem statement
theorem hcl_production (nacl_moles hno3_moles : ℕ) 
  (h : nacl_moles = 3 ∧ hno3_moles = 3) : 
  ∃ (r : Reaction), r.reactant1 = nacl_moles ∧ r.reactant2 = hno3_moles ∧ 
  stoichiometric_ratio r ∧ r.product = 3 :=
sorry

end NUMINAMATH_CALUDE_hcl_production_l3150_315072


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3150_315034

theorem sine_cosine_sum_equals_one : 
  Real.sin (π / 2 + π / 3) + Real.cos (π / 2 - π / 6) = 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3150_315034


namespace NUMINAMATH_CALUDE_ratio_of_sums_l3150_315040

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  firstTerm : ℕ
  difference : ℕ
  length : ℕ

/-- Calculates the sum of an arithmetic progression -/
def sumArithmeticProgression (ap : ArithmeticProgression) : ℕ :=
  ap.length * (2 * ap.firstTerm + (ap.length - 1) * ap.difference) / 2

/-- Generates a list of arithmetic progressions for the first group -/
def firstGroup : List ArithmeticProgression :=
  List.range 15 |>.map (fun i => ⟨i + 1, 2 * (i + 1), 10⟩)

/-- Generates a list of arithmetic progressions for the second group -/
def secondGroup : List ArithmeticProgression :=
  List.range 15 |>.map (fun i => ⟨i + 1, 2 * i + 1, 10⟩)

/-- Calculates the sum of all elements in a group of arithmetic progressions -/
def sumGroup (group : List ArithmeticProgression) : ℕ :=
  group.map sumArithmeticProgression |>.sum

theorem ratio_of_sums : (sumGroup firstGroup : ℚ) / (sumGroup secondGroup) = 160 / 151 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l3150_315040


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3150_315052

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 1485) :
  a * b = 17820 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3150_315052


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3150_315029

theorem arithmetic_sequence_length :
  ∀ (a₁ d : ℤ) (n : ℕ),
    a₁ = -6 →
    d = 5 →
    a₁ + (n - 1) * d = 59 →
    n = 14 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3150_315029


namespace NUMINAMATH_CALUDE_rural_school_absence_percentage_l3150_315051

/-- Rural School Z student absence problem -/
theorem rural_school_absence_percentage :
  let total_students : ℕ := 150
  let boys : ℕ := 90
  let girls : ℕ := 60
  let boys_absent_ratio : ℚ := 1 / 9
  let girls_absent_ratio : ℚ := 1 / 4
  let absent_students : ℚ := boys_absent_ratio * boys + girls_absent_ratio * girls
  absent_students / total_students = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rural_school_absence_percentage_l3150_315051


namespace NUMINAMATH_CALUDE_quadricycle_count_l3150_315044

theorem quadricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 30) : ∃ (b t q : ℕ), 
  b + t + q = total_children ∧ 
  2 * b + 3 * t + 4 * q = total_wheels ∧ 
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadricycle_count_l3150_315044


namespace NUMINAMATH_CALUDE_total_checks_purchased_l3150_315092

/-- Represents the number of travelers checks purchased -/
structure TravelersChecks where
  fifty : ℕ    -- number of $50 checks
  hundred : ℕ  -- number of $100 checks

/-- The total value of all travelers checks -/
def total_value (tc : TravelersChecks) : ℕ :=
  50 * tc.fifty + 100 * tc.hundred

/-- The average value of remaining checks after spending 6 $50 checks -/
def average_remaining (tc : TravelersChecks) : ℚ :=
  (total_value tc - 300) / (tc.fifty + tc.hundred - 6 : ℚ)

/-- Theorem stating the total number of travelers checks purchased -/
theorem total_checks_purchased :
  ∃ (tc : TravelersChecks),
    total_value tc = 1800 ∧
    average_remaining tc = 62.5 ∧
    tc.fifty + tc.hundred = 33 :=
  sorry

end NUMINAMATH_CALUDE_total_checks_purchased_l3150_315092


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l3150_315024

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l3150_315024


namespace NUMINAMATH_CALUDE_part_one_part_two_l3150_315008

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k + 1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * m * f a 0 x

-- Theorem for part (1)
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) → k = 0 :=
sorry

-- Theorem for part (2)
theorem part_two (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 0 1 = 3/2) 
  (h4 : ∀ x : ℝ, x ≥ 0 → g a m x ≥ -6) 
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ g a m x = -6) :
  m = 2 * Real.sqrt 2 ∨ m = -2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l3150_315008


namespace NUMINAMATH_CALUDE_barn_paint_area_l3150_315041

/-- Represents the dimensions of a rectangular prism -/
structure BarnDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in the barn -/
def total_paint_area (dim : BarnDimensions) : ℝ :=
  let end_wall_area := 2 * dim.width * dim.height
  let side_wall_area := 2 * dim.length * dim.height
  let ceiling_area := dim.length * dim.width
  let partition_area := 2 * dim.length * dim.height
  2 * (end_wall_area + side_wall_area) + ceiling_area + partition_area

/-- The barn dimensions -/
def barn : BarnDimensions :=
  { length := 15
  , width := 12
  , height := 6 }

theorem barn_paint_area :
  total_paint_area barn = 1008 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l3150_315041
