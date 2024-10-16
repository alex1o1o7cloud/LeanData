import Mathlib

namespace NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l2357_235711

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, f a y ≥ f a x) → a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l2357_235711


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l2357_235774

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l2357_235774


namespace NUMINAMATH_CALUDE_melanie_brownies_given_out_l2357_235782

def total_brownies : ℕ := 12 * 25

def bake_sale_brownies : ℕ := (7 * total_brownies) / 10

def remaining_after_bake_sale : ℕ := total_brownies - bake_sale_brownies

def container_brownies : ℕ := (2 * remaining_after_bake_sale) / 3

def remaining_after_container : ℕ := remaining_after_bake_sale - container_brownies

def charity_brownies : ℕ := (2 * remaining_after_container) / 5

def brownies_given_out : ℕ := remaining_after_container - charity_brownies

theorem melanie_brownies_given_out : brownies_given_out = 18 := by
  sorry

end NUMINAMATH_CALUDE_melanie_brownies_given_out_l2357_235782


namespace NUMINAMATH_CALUDE_total_sacks_needed_l2357_235718

/-- The number of sacks of strawberries needed for the first bakery per week -/
def bakery1_weekly_need : ℕ := 2

/-- The number of sacks of strawberries needed for the second bakery per week -/
def bakery2_weekly_need : ℕ := 4

/-- The number of sacks of strawberries needed for the third bakery per week -/
def bakery3_weekly_need : ℕ := 12

/-- The number of weeks for which the supply is calculated -/
def supply_period : ℕ := 4

/-- Theorem stating that the total number of sacks needed for all bakeries in 4 weeks is 72 -/
theorem total_sacks_needed :
  (bakery1_weekly_need + bakery2_weekly_need + bakery3_weekly_need) * supply_period = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_sacks_needed_l2357_235718


namespace NUMINAMATH_CALUDE_multinomial_binomial_equality_l2357_235766

theorem multinomial_binomial_equality (n : ℕ) : 
  Nat.choose n 2 * Nat.choose (n - 2) 2 = 3 * Nat.choose n 4 + 3 * Nat.choose n 3 := by
  sorry

end NUMINAMATH_CALUDE_multinomial_binomial_equality_l2357_235766


namespace NUMINAMATH_CALUDE_toms_profit_is_21988_l2357_235773

/-- Calculates Tom's profit from making the world's largest dough ball -/
def toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                (salt_needed : ℕ) (salt_cost : ℚ)
                (sugar_needed : ℕ) (sugar_cost : ℚ)
                (butter_needed : ℕ) (butter_cost : ℕ)
                (chef_cost : ℕ) (promotion_cost : ℕ)
                (ticket_price : ℕ) (tickets_sold : ℕ) : ℤ :=
  let flour_cost := (flour_needed / flour_bag_size) * flour_bag_cost
  let salt_cost_total := (salt_needed : ℚ) * salt_cost
  let sugar_cost_total := (sugar_needed : ℚ) * sugar_cost
  let butter_cost_total := butter_needed * butter_cost
  let total_cost := flour_cost + salt_cost_total.ceil + sugar_cost_total.ceil + 
                    butter_cost_total + chef_cost + promotion_cost
  let revenue := ticket_price * tickets_sold
  revenue - total_cost

/-- Tom's profit from making the world's largest dough ball is $21988 -/
theorem toms_profit_is_21988 : 
  toms_profit 500 50 20 10 (2/10) 20 (1/2) 50 2 700 1000 20 1200 = 21988 := by
  sorry

end NUMINAMATH_CALUDE_toms_profit_is_21988_l2357_235773


namespace NUMINAMATH_CALUDE_hike_return_pace_l2357_235722

theorem hike_return_pace 
  (total_distance : ℝ) 
  (outbound_pace : ℝ) 
  (total_time : ℝ) 
  (return_pace : ℝ) :
  total_distance = 12 →
  outbound_pace = 4 →
  total_time = 5 →
  total_distance / outbound_pace + total_distance / return_pace = total_time →
  return_pace = 6 :=
by sorry

end NUMINAMATH_CALUDE_hike_return_pace_l2357_235722


namespace NUMINAMATH_CALUDE_intersection_A_B_l2357_235708

def A : Set ℤ := {x | |x| < 3}
def B : Set ℤ := {x | |x| > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2357_235708


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2357_235750

/-- The quadratic function f(x) = x^2 + 2px + p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 2*p*x + p^2

theorem quadratic_minimum (p : ℝ) (hp : p > 0) (hp2 : 2*p + p^2 = 10) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p x_min ≤ f p x ∧ x_min = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2357_235750


namespace NUMINAMATH_CALUDE_correct_calculation_l2357_235724

theorem correct_calculation (x : ℝ) : (x + 20) * 5 = 225 → (x + 20) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2357_235724


namespace NUMINAMATH_CALUDE_households_with_car_l2357_235769

theorem households_with_car (total : ℕ) (without_car_or_bike : ℕ) (with_both : ℕ) (with_bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 16)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 60 := by
  sorry

#check households_with_car

end NUMINAMATH_CALUDE_households_with_car_l2357_235769


namespace NUMINAMATH_CALUDE_complex_sum_l2357_235768

theorem complex_sum (z : ℂ) (h : z = (1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2)) :
  z + 2*z^2 + 3*z^3 + 4*z^4 + 5*z^5 + 6*z^6 = 3 - Complex.I * 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_l2357_235768


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2357_235786

def running_shoes_original_price : ℝ := 80
def casual_shoes_original_price : ℝ := 60
def running_shoes_discount : ℝ := 0.25
def casual_shoes_discount : ℝ := 0.40
def sales_tax_rate : ℝ := 0.08
def num_running_shoes : ℕ := 2
def num_casual_shoes : ℕ := 3

def total_cost : ℝ :=
  let running_shoes_discounted_price := running_shoes_original_price * (1 - running_shoes_discount)
  let casual_shoes_discounted_price := casual_shoes_original_price * (1 - casual_shoes_discount)
  let subtotal := num_running_shoes * running_shoes_discounted_price + num_casual_shoes * casual_shoes_discounted_price
  subtotal * (1 + sales_tax_rate)

theorem total_cost_is_correct : total_cost = 246.24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2357_235786


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l2357_235783

theorem magnitude_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l2357_235783


namespace NUMINAMATH_CALUDE_pedro_extra_squares_l2357_235761

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_extra_squares_l2357_235761


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2357_235723

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 7*x - 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 11*x + 24 = (x + b) * (x + c)) →
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2357_235723


namespace NUMINAMATH_CALUDE_triplet_characterization_l2357_235748

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ (a * b * c - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {(3, 5, 15), (3, 15, 5), (2, 4, 8), (2, 8, 4), (2, 2, 4), (2, 4, 2), (2, 2, 2)}

theorem triplet_characterization :
  {(a, b, c) | is_valid_triplet a b c} = valid_triplets :=
by sorry

end NUMINAMATH_CALUDE_triplet_characterization_l2357_235748


namespace NUMINAMATH_CALUDE_role_assignment_theorem_l2357_235739

def number_of_ways_to_assign_roles (men : Nat) (women : Nat) (male_roles : Nat) (female_roles : Nat) (either_roles : Nat) : Nat :=
  -- Number of ways to assign male roles
  (men.choose male_roles) * (male_roles.factorial) *
  -- Number of ways to assign female roles
  (women.choose female_roles) * (female_roles.factorial) *
  -- Number of ways to assign either-gender roles
  ((men + women - male_roles - female_roles).choose either_roles) * (either_roles.factorial)

theorem role_assignment_theorem :
  number_of_ways_to_assign_roles 6 7 3 3 2 = 1058400 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_theorem_l2357_235739


namespace NUMINAMATH_CALUDE_cone_surface_area_l2357_235788

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π := by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2357_235788


namespace NUMINAMATH_CALUDE_fraction_filled_equals_half_l2357_235709

/-- Represents the fraction of a cistern that can be filled in 15 minutes -/
def fraction_filled_in_15_min : ℚ := 1 / 2

/-- The time it takes to fill half of the cistern -/
def time_to_fill_half : ℕ := 15

/-- Theorem stating that the fraction of the cistern filled in 15 minutes is 1/2 -/
theorem fraction_filled_equals_half : 
  fraction_filled_in_15_min = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_filled_equals_half_l2357_235709


namespace NUMINAMATH_CALUDE_shortest_chord_equation_l2357_235725

-- Define the line l
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 3 + t ∧ p.2 = 1 + a * t}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for shortest chord
def shortest_chord (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
  ∀ X Y : ℝ × ℝ, X ∈ l ∧ X ∈ C ∧ Y ∈ l ∧ Y ∈ C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Theorem statement
theorem shortest_chord_equation :
  ∃ a : ℝ, shortest_chord (line_l a) circle_C →
  ∀ p : ℝ × ℝ, p ∈ line_l a ↔ p.1 + p.2 = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_equation_l2357_235725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2357_235771

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 = 8) : 
  a 2 - a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2357_235771


namespace NUMINAMATH_CALUDE_in_class_calculation_l2357_235737

theorem in_class_calculation :
  (((4.2 : ℝ) + 2.2) / 0.08 = 80) ∧
  (100 / 0.4 / 2.5 = 100) := by
  sorry

end NUMINAMATH_CALUDE_in_class_calculation_l2357_235737


namespace NUMINAMATH_CALUDE_seed_to_sprout_probability_is_correct_l2357_235716

/-- The germination rate of a batch of seeds -/
def germination_rate : ℝ := 0.9

/-- The survival rate of sprouts after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a sprout -/
def seed_to_sprout_probability : ℝ := germination_rate * survival_rate

/-- Theorem: The probability that a randomly selected seed will grow into a sprout is 0.72 -/
theorem seed_to_sprout_probability_is_correct : seed_to_sprout_probability = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_seed_to_sprout_probability_is_correct_l2357_235716


namespace NUMINAMATH_CALUDE_smallest_k_satisfying_inequality_l2357_235707

theorem smallest_k_satisfying_inequality (n m : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) :
  (∀ k : ℕ, k % 3 = 0 → (64^k + 32^m > 4^(16 + n^2) → k ≥ 6)) ∧
  (64^6 + 32^m > 4^(16 + n^2)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_satisfying_inequality_l2357_235707


namespace NUMINAMATH_CALUDE_min_value_theorem_l2357_235775

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2357_235775


namespace NUMINAMATH_CALUDE_function_equality_l2357_235746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else a^x

theorem function_equality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (Real.exp 2) = f a (-2) → a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2357_235746


namespace NUMINAMATH_CALUDE_triangle_inequality_l2357_235787

/-- For any triangle with sides a, b, c and area S, 
    the inequality a^2 + b^2 + c^2 - 1/2(|a-b| + |b-c| + |c-a|)^2 ≥ 4√3 S holds. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 - 1/2 * (|a - b| + |b - c| + |c - a|)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2357_235787


namespace NUMINAMATH_CALUDE_dress_designs_count_l2357_235759

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options available -/
def num_sleeve_lengths : ℕ := 2

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

theorem dress_designs_count : total_designs = 24 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l2357_235759


namespace NUMINAMATH_CALUDE_intersection_A_B_l2357_235731

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ t : ℝ, x = t^2 + 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x * (x - 1) = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2357_235731


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l2357_235762

theorem modular_congruence_existence (a c : ℕ) (b : ℤ) :
  ∃ x : ℕ, (a ^ x + x : ℤ) ≡ b [ZMOD c] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l2357_235762


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_inequality_l2357_235765

theorem triangle_sine_cosine_inequality (A B C : ℝ) (h : A + B + C = π) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_inequality_l2357_235765


namespace NUMINAMATH_CALUDE_trip_duration_is_eight_hours_l2357_235754

/-- Represents a car trip with varying speeds -/
structure CarTrip where
  initial_hours : ℝ
  initial_speed : ℝ
  additional_speed : ℝ
  average_speed : ℝ

/-- Calculates the total duration of the car trip -/
def trip_duration (trip : CarTrip) : ℝ :=
  sorry

/-- Theorem stating that the trip duration is 8 hours given the specific conditions -/
theorem trip_duration_is_eight_hours (trip : CarTrip) 
  (h1 : trip.initial_hours = 4)
  (h2 : trip.initial_speed = 50)
  (h3 : trip.additional_speed = 80)
  (h4 : trip.average_speed = 65) :
  trip_duration trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_is_eight_hours_l2357_235754


namespace NUMINAMATH_CALUDE_money_loses_exchange_value_on_deserted_island_l2357_235760

-- Define the basic concepts
def Person : Type := String
def Money : Type := ℕ
def Item : Type := String

-- Define the properties of money
structure MoneyProperties :=
  (medium_of_exchange : Bool)
  (store_of_value : Bool)
  (unit_of_account : Bool)
  (standard_of_deferred_payment : Bool)

-- Define the island environment
structure Island :=
  (inhabitants : List Person)
  (items : List Item)
  (currency : Money)

-- Define the value of money in a given context
def money_value (island : Island) (props : MoneyProperties) : ℝ := 
  sorry

-- Theorem: Money loses its value as a medium of exchange on a deserted island
theorem money_loses_exchange_value_on_deserted_island 
  (island : Island) 
  (props : MoneyProperties) :
  island.inhabitants.length = 1 →
  money_value island props = 0 :=
sorry

end NUMINAMATH_CALUDE_money_loses_exchange_value_on_deserted_island_l2357_235760


namespace NUMINAMATH_CALUDE_group_collection_theorem_l2357_235703

/-- Calculates the total collection amount in rupees for a group of students -/
def totalCollectionInRupees (groupSize : ℕ) : ℚ :=
  (groupSize * groupSize : ℚ) / 100

/-- Theorem: The total collection amount for a group of 45 students is 20.25 rupees -/
theorem group_collection_theorem :
  totalCollectionInRupees 45 = 20.25 := by
  sorry

#eval totalCollectionInRupees 45

end NUMINAMATH_CALUDE_group_collection_theorem_l2357_235703


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l2357_235735

-- Define the loan parameters
def initial_loan : ℝ := 10000
def initial_period : ℝ := 2
def additional_loan : ℝ := 12000
def additional_period : ℝ := 3
def total_repayment : ℝ := 27160

-- Define the function to calculate the total amount to be repaid
def total_amount (rate : ℝ) : ℝ :=
  initial_loan * (1 + rate * (initial_period + additional_period)) +
  additional_loan * (1 + rate * additional_period)

-- Theorem statement
theorem interest_rate_is_six_percent :
  ∃ (rate : ℝ), rate > 0 ∧ rate < 1 ∧ total_amount rate = total_repayment ∧ rate = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l2357_235735


namespace NUMINAMATH_CALUDE_standard_deck_three_card_selections_l2357_235702

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cardsPerSuit : Nat)
  (redSuits : Nat)
  (blackSuits : Nat)

/-- A standard deck of 52 cards -/
def standardDeck : Deck :=
  { cards := 52
  , suits := 4
  , cardsPerSuit := 13
  , redSuits := 2
  , blackSuits := 2 }

/-- The number of ways to select three different cards from a deck, where order matters -/
def threeCardSelections (d : Deck) : Nat :=
  d.cards * (d.cards - 1) * (d.cards - 2)

/-- Theorem stating the number of ways to select three different cards from a standard deck -/
theorem standard_deck_three_card_selections :
  threeCardSelections standardDeck = 132600 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_three_card_selections_l2357_235702


namespace NUMINAMATH_CALUDE_B_equals_A_l2357_235712

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x ∈ A}

theorem B_equals_A : B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_A_l2357_235712


namespace NUMINAMATH_CALUDE_f_of_2_equals_neg_1_l2357_235749

-- Define the function f
def f : ℝ → ℝ := λ x => (x - 1)^2 - 2*(x - 1)

-- State the theorem
theorem f_of_2_equals_neg_1 : f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_neg_1_l2357_235749


namespace NUMINAMATH_CALUDE_symmetry_implies_k_and_b_l2357_235714

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are symmetric with respect to the vertical line x = a -/
def symmetric_lines (l1 l2 : Line) (a : ℝ) : Prop :=
  l1.slope = -l2.slope ∧
  l1.intercept + l2.intercept = 2 * (l1.slope * a + l1.intercept)

/-- The main theorem stating the conditions for symmetry and the resulting values of k and b -/
theorem symmetry_implies_k_and_b (k b : ℝ) :
  symmetric_lines (Line.mk k 3) (Line.mk 2 b) 1 →
  k = -2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_k_and_b_l2357_235714


namespace NUMINAMATH_CALUDE_sum_of_root_products_l2357_235798

theorem sum_of_root_products (a b c d : ℂ) : 
  (2 * a^4 - 6 * a^3 + 14 * a^2 - 13 * a + 8 = 0) →
  (2 * b^4 - 6 * b^3 + 14 * b^2 - 13 * b + 8 = 0) →
  (2 * c^4 - 6 * c^3 + 14 * c^2 - 13 * c + 8 = 0) →
  (2 * d^4 - 6 * d^3 + 14 * d^2 - 13 * d + 8 = 0) →
  a * b + a * c + a * d + b * c + b * d + c * d = -7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l2357_235798


namespace NUMINAMATH_CALUDE_problem_statement_l2357_235770

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 + b^2 ≥ 1/2) ∧ (a*b ≤ 1/4) ∧ (1/a + 1/b > 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2357_235770


namespace NUMINAMATH_CALUDE_allan_balloons_l2357_235719

def park_balloon_problem (jake_initial : ℕ) (jake_bought : ℕ) (difference : ℕ) : ℕ :=
  let jake_total := jake_initial + jake_bought
  jake_total - difference

theorem allan_balloons :
  park_balloon_problem 3 4 1 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_allan_balloons_l2357_235719


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2357_235753

theorem solution_set_inequality (x : ℝ) :
  (2*x - 3) * (x + 1) < 0 ↔ -1 < x ∧ x < 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2357_235753


namespace NUMINAMATH_CALUDE_tangent_line_at_2_min_value_in_interval_l2357_235729

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 2 = f' 2 * (x - 2) :=
sorry

-- Theorem for the minimum value in the interval [-3, 3]
theorem min_value_in_interval : 
  ∃ x₀ ∈ Set.Icc (-3 : ℝ) 3, ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x₀ ≤ f x ∧ f x₀ = -17 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_min_value_in_interval_l2357_235729


namespace NUMINAMATH_CALUDE_tina_postcard_price_l2357_235730

/-- Proves that the price per postcard is $5, given the conditions of Tina's postcard sales. -/
theorem tina_postcard_price :
  let postcards_per_day : ℕ := 30
  let days_sold : ℕ := 6
  let total_earned : ℕ := 900
  let total_postcards : ℕ := postcards_per_day * days_sold
  let price_per_postcard : ℚ := total_earned / total_postcards
  price_per_postcard = 5 := by sorry

end NUMINAMATH_CALUDE_tina_postcard_price_l2357_235730


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l2357_235701

theorem smallest_three_digit_multiple_plus_one : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∃ k : ℕ, n = 8 * k + 1) ∧
  (∀ m : ℕ, m < n →
    ¬(100 ≤ m ∧ m < 1000 ∧
      (∃ k : ℕ, m = 3 * k + 1) ∧
      (∃ k : ℕ, m = 4 * k + 1) ∧
      (∃ k : ℕ, m = 5 * k + 1) ∧
      (∃ k : ℕ, m = 7 * k + 1) ∧
      (∃ k : ℕ, m = 8 * k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l2357_235701


namespace NUMINAMATH_CALUDE_sequence_properties_l2357_235767

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1 / (2 * n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 1 / (2 * sequence_a (n + 1)) = 1 / (2 * sequence_a n) + 1) →
  (∀ n : ℕ, n > 0 → 1 / sequence_a (n + 1) - 1 / sequence_a n = 2) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 1 / (2 * n - 1)) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) = n / (2 * n + 1)) ∧
  (∀ n : ℕ, n > 0 → 
    ((Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) > 16 / 33) ↔ n > 16) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2357_235767


namespace NUMINAMATH_CALUDE_equation_solution_l2357_235743

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0 ∧ x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2357_235743


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2357_235758

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0, 1}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2357_235758


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l2357_235713

/-- Proves that the cost price of an item is 6672 when the selling price is 8340
    and the markup is 25%. -/
theorem furniture_shop_cost_price : 
  ∀ (cost_price selling_price : ℝ),
  selling_price = 8340 →
  selling_price = cost_price * (1 + 0.25) →
  cost_price = 6672 := by sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l2357_235713


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2357_235742

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- Square inscribed with one vertex at the right angle -/
def corner_square (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b

/-- Square inscribed with one side along the hypotenuse -/
def hypotenuse_square (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (hx : corner_square t x) (hy : hypotenuse_square t y) : 
  x / y = 1 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2357_235742


namespace NUMINAMATH_CALUDE_sqrt_eight_between_two_and_three_l2357_235781

theorem sqrt_eight_between_two_and_three : 2 < Real.sqrt 8 ∧ Real.sqrt 8 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_between_two_and_three_l2357_235781


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_property_l2357_235721

/-- A right-angled triangular pyramid with right-angle face areas S₁, S₂, S₃ and oblique face area S -/
structure RightTriangularPyramid where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S : ℝ
  S₁_pos : 0 < S₁
  S₂_pos : 0 < S₂
  S₃_pos : 0 < S₃
  S_pos : 0 < S

/-- The property of a right-angled triangular pyramid -/
theorem right_triangular_pyramid_property (p : RightTriangularPyramid) : 
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.S^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_property_l2357_235721


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2357_235728

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: Intersection of A and B when a = -2
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x | -5 ≤ x ∧ x < -1} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2357_235728


namespace NUMINAMATH_CALUDE_smallest_d_value_l2357_235791

theorem smallest_d_value (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h : ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3*d*x^2 + 3*x + e^3) :
  d ≥ 1 ∧ ∃ (a₀ b₀ c₀ e₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ e₀ > 0 ∧
    (∀ x : ℝ, (x + a₀) * (x + b₀) * (x + c₀) = x^3 + 3*x^2 + 3*x + e₀^3) := by
  sorry

#check smallest_d_value

end NUMINAMATH_CALUDE_smallest_d_value_l2357_235791


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l2357_235796

-- Define the number of red and green shoes
def num_red_shoes : ℕ := 7
def num_green_shoes : ℕ := 3

-- Define the total number of shoes
def total_shoes : ℕ := num_red_shoes + num_green_shoes

-- Define the number of shoes to be drawn
def shoes_drawn : ℕ := 2

-- Define the probability of drawing two red shoes
def prob_two_red_shoes : ℚ := 7 / 15

-- Theorem statement
theorem probability_two_red_shoes :
  (Nat.choose num_red_shoes shoes_drawn : ℚ) / (Nat.choose total_shoes shoes_drawn : ℚ) = prob_two_red_shoes :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l2357_235796


namespace NUMINAMATH_CALUDE_trumpet_to_running_ratio_l2357_235794

/-- Proves that the ratio of time spent practicing trumpet to time spent running is 2:1 -/
theorem trumpet_to_running_ratio 
  (basketball_time : ℕ) 
  (trumpet_time : ℕ) 
  (h1 : basketball_time = 10)
  (h2 : trumpet_time = 40) :
  (trumpet_time : ℚ) / (2 * basketball_time) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_trumpet_to_running_ratio_l2357_235794


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2357_235756

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 + 9 * x + 6 ≤ 0

theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2357_235756


namespace NUMINAMATH_CALUDE_alpha_sum_sixth_power_l2357_235755

theorem alpha_sum_sixth_power (α₁ α₂ α₃ : ℂ) 
  (sum_zero : α₁ + α₂ + α₃ = 0)
  (sum_squares : α₁^2 + α₂^2 + α₃^2 = 2)
  (sum_cubes : α₁^3 + α₂^3 + α₃^3 = 4) :
  α₁^6 + α₂^6 + α₃^6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_alpha_sum_sixth_power_l2357_235755


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l2357_235764

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 3 * a 13 = 16) :
  a 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l2357_235764


namespace NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l2357_235780

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (sigFigs : ℕ) : ScientificNotation :=
  sorry

theorem bird_nest_area_scientific_notation :
  let area : ℝ := 258000
  let scientific_form := toScientificNotation area
  let rounded_form := roundToSignificantFigures scientific_form 2
  rounded_form.coefficient = 2.6 ∧ rounded_form.exponent = 5 :=
sorry

end NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l2357_235780


namespace NUMINAMATH_CALUDE_route_comparison_l2357_235772

/-- Represents the time difference between two routes when all lights are red on the first route -/
def route_time_difference (first_route_base_time : ℕ) (red_light_delay : ℕ) (num_lights : ℕ) (second_route_time : ℕ) : ℕ :=
  (first_route_base_time + red_light_delay * num_lights) - second_route_time

theorem route_comparison :
  route_time_difference 10 3 3 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_route_comparison_l2357_235772


namespace NUMINAMATH_CALUDE_billy_is_48_l2357_235733

-- Define Billy's age and Joe's age
def billy_age : ℕ := sorry
def joe_age : ℕ := sorry

-- State the conditions
axiom age_relation : billy_age = 3 * joe_age
axiom age_sum : billy_age + joe_age = 64

-- Theorem to prove
theorem billy_is_48 : billy_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_billy_is_48_l2357_235733


namespace NUMINAMATH_CALUDE_remainder_eleven_pow_2023_mod_8_l2357_235792

theorem remainder_eleven_pow_2023_mod_8 : 11^2023 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_pow_2023_mod_8_l2357_235792


namespace NUMINAMATH_CALUDE_paving_cost_theorem_l2357_235747

/-- Represents the dimensions and cost of a rectangular room -/
structure RectangularRoom where
  length : ℝ
  width : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a triangular room -/
structure TriangularRoom where
  base : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a trapezoidal room -/
structure TrapezoidalRoom where
  parallel_side1 : ℝ
  parallel_side2 : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of paving three rooms -/
def total_paving_cost (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom) : ℝ :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  (0.5 * room2.base * room2.height * room2.cost_per_sqm) +
  (0.5 * (room3.parallel_side1 + room3.parallel_side2) * room3.height * room3.cost_per_sqm)

/-- Theorem stating the total cost of paving the three rooms -/
theorem paving_cost_theorem (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 1400 })
  (h2 : room2 = { base := 4, height := 3, cost_per_sqm := 1500 })
  (h3 : room3 = { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }) :
  total_paving_cost room1 room2 room3 = 56875 := by
  sorry

#eval total_paving_cost
  { length := 5.5, width := 3.75, cost_per_sqm := 1400 }
  { base := 4, height := 3, cost_per_sqm := 1500 }
  { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }

end NUMINAMATH_CALUDE_paving_cost_theorem_l2357_235747


namespace NUMINAMATH_CALUDE_complex_square_equality_l2357_235710

theorem complex_square_equality (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (a : ℂ) + (b : ℂ) * Complex.I = 4 + 3 * Complex.I ↔ 
  ((a : ℂ) + (b : ℂ) * Complex.I) ^ 2 = 7 + 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2357_235710


namespace NUMINAMATH_CALUDE_popsicle_sticks_left_l2357_235741

def total_budget : ℕ := 10
def freezer_capacity : ℕ := 30
def mold_cost : ℕ := 3
def mold_capacity : ℕ := 10
def stick_pack_cost : ℕ := 1
def stick_pack_quantity : ℕ := 100
def orange_juice_cost : ℕ := 2
def orange_juice_popsicles : ℕ := 20
def apple_juice_cost : ℕ := 3
def apple_juice_popsicles : ℕ := 30
def grape_juice_cost : ℕ := 4
def grape_juice_popsicles : ℕ := 40

theorem popsicle_sticks_left : ℕ := by
  sorry

#check popsicle_sticks_left = 70

end NUMINAMATH_CALUDE_popsicle_sticks_left_l2357_235741


namespace NUMINAMATH_CALUDE_equality_condition_l2357_235706

theorem equality_condition (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 3 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2357_235706


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2357_235744

theorem power_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2357_235744


namespace NUMINAMATH_CALUDE_some_ounce_glass_size_l2357_235704

/-- Given the following conditions:
  - Claudia has 122 ounces of water
  - She fills six 5-ounce glasses and four 8-ounce glasses
  - She can fill 15 glasses of the some-ounce size with the remaining water
  Prove that the size of the some-ounce glasses is 4 ounces. -/
theorem some_ounce_glass_size (total_water : ℕ) (five_ounce_count : ℕ) (eight_ounce_count : ℕ) (some_ounce_count : ℕ)
  (h1 : total_water = 122)
  (h2 : five_ounce_count = 6)
  (h3 : eight_ounce_count = 4)
  (h4 : some_ounce_count = 15)
  (h5 : total_water = 5 * five_ounce_count + 8 * eight_ounce_count + some_ounce_count * (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count) :
  (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_ounce_glass_size_l2357_235704


namespace NUMINAMATH_CALUDE_det_roots_cubic_l2357_235738

theorem det_roots_cubic (p q r a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix := !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c]
  Matrix.det matrix = r + 2*q + 4*p + 4 := by
  sorry

end NUMINAMATH_CALUDE_det_roots_cubic_l2357_235738


namespace NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2357_235705

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![1, 3]
def c (t : ℝ) : Fin 2 → ℝ := ![t, 2]

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, (∀ i : Fin 2, (a i - c t i) * b i = 0) → t = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2357_235705


namespace NUMINAMATH_CALUDE_bus_departure_interval_l2357_235715

/-- Represents the time interval between bus departures -/
def bus_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) : ℚ :=
  total_time / ((num_buses - 1) * minutes_per_hour)

theorem bus_departure_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) 
  (h1 : total_time = 60) 
  (h2 : minutes_per_hour = 60)
  (h3 : num_buses = 11) :
  bus_interval total_time minutes_per_hour num_buses = 6 := by
  sorry

#eval bus_interval 60 60 11

end NUMINAMATH_CALUDE_bus_departure_interval_l2357_235715


namespace NUMINAMATH_CALUDE_wall_length_calculation_l2357_235790

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side * mirror_side) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l2357_235790


namespace NUMINAMATH_CALUDE_jiangsu_population_scientific_notation_l2357_235727

/-- The population of Jiangsu Province in 2021 -/
def jiangsu_population : ℕ := 85000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Theorem: The population of Jiangsu Province expressed in scientific notation -/
theorem jiangsu_population_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.significand * (10 : ℝ) ^ sn.exponent) = jiangsu_population := by
  sorry

end NUMINAMATH_CALUDE_jiangsu_population_scientific_notation_l2357_235727


namespace NUMINAMATH_CALUDE_max_value_expression_l2357_235763

theorem max_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2) / b^2 ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2357_235763


namespace NUMINAMATH_CALUDE_interest_difference_approx_l2357_235720

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  |simple_interest principal simple_rate time - compound_interest principal compound_rate time|

theorem interest_difference_approx :
  ∃ ε > 0, |interest_difference 10000 0.04 0.06 12 - 1189| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l2357_235720


namespace NUMINAMATH_CALUDE_real_solutions_range_l2357_235778

theorem real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_range_l2357_235778


namespace NUMINAMATH_CALUDE_gina_collected_two_bags_l2357_235777

/-- The number of bags Gina collected by herself -/
def gina_bags : ℕ := 2

/-- The number of bags collected by the rest of the neighborhood -/
def neighborhood_bags : ℕ := 82 * gina_bags

/-- The weight of each bag in pounds -/
def bag_weight : ℕ := 4

/-- The total weight of litter collected in pounds -/
def total_weight : ℕ := 664

/-- Theorem stating that Gina collected 2 bags of litter -/
theorem gina_collected_two_bags :
  gina_bags = 2 ∧
  neighborhood_bags = 82 * gina_bags ∧
  bag_weight = 4 ∧
  total_weight = 664 ∧
  total_weight = bag_weight * (gina_bags + neighborhood_bags) :=
by sorry

end NUMINAMATH_CALUDE_gina_collected_two_bags_l2357_235777


namespace NUMINAMATH_CALUDE_new_years_day_in_big_month_l2357_235795

-- Define the set of months
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

-- Define the set of holidays
inductive Holiday
| NewYearsDay
| ChildrensDay
| TeachersDay

-- Define a function to get the month of a holiday
def holiday_month (h : Holiday) : Month :=
  match h with
  | Holiday.NewYearsDay => Month.January
  | Holiday.ChildrensDay => Month.June
  | Holiday.TeachersDay => Month.September

-- Define the set of big months
def is_big_month (m : Month) : Prop :=
  m = Month.January ∨ m = Month.March ∨ m = Month.May ∨
  m = Month.July ∨ m = Month.August ∨ m = Month.October ∨
  m = Month.December

-- Theorem: New Year's Day falls in a big month
theorem new_years_day_in_big_month :
  is_big_month (holiday_month Holiday.NewYearsDay) :=
by sorry

end NUMINAMATH_CALUDE_new_years_day_in_big_month_l2357_235795


namespace NUMINAMATH_CALUDE_pencil_length_theorem_l2357_235776

def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

theorem pencil_length_theorem (original_length sharpened_off : ℕ) 
  (h1 : original_length = 31)
  (h2 : sharpened_off = 17) :
  pencil_length_after_sharpening original_length sharpened_off = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_theorem_l2357_235776


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2357_235745

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2357_235745


namespace NUMINAMATH_CALUDE_problem1_simplification_l2357_235799

theorem problem1_simplification (x y : ℝ) : 
  y * (4 * x - 3 * y) + (x - 2 * y)^2 = x^2 + y^2 := by sorry

end NUMINAMATH_CALUDE_problem1_simplification_l2357_235799


namespace NUMINAMATH_CALUDE_fly_distance_l2357_235751

/-- The distance traveled by a fly between two runners --/
theorem fly_distance (joe_speed maria_speed fly_speed initial_distance : ℝ) :
  joe_speed = 10 ∧ 
  maria_speed = 8 ∧ 
  fly_speed = 15 ∧ 
  initial_distance = 3 →
  (fly_speed * initial_distance) / (joe_speed + maria_speed) = 5/2 := by
  sorry

#check fly_distance

end NUMINAMATH_CALUDE_fly_distance_l2357_235751


namespace NUMINAMATH_CALUDE_telephone_fee_properties_l2357_235779

-- Define the telephone fee function
def telephone_fee (x : ℝ) : ℝ := 0.4 * x + 18

-- Theorem statement
theorem telephone_fee_properties :
  (∀ x : ℝ, telephone_fee x = 0.4 * x + 18) ∧
  (telephone_fee 10 = 22) ∧
  (telephone_fee 20 = 26) := by
  sorry


end NUMINAMATH_CALUDE_telephone_fee_properties_l2357_235779


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_l2357_235785

theorem integer_list_mean_mode (x : ℕ) : 
  x ≤ 120 →
  x > 0 →
  let list := [45, 76, 110, x, x]
  (list.sum / list.length : ℚ) = 2 * x →
  x = 29 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_l2357_235785


namespace NUMINAMATH_CALUDE_girls_combined_average_l2357_235726

structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

def central : School := { boys_score := 68, girls_score := 72, combined_score := 70 }
def delta : School := { boys_score := 78, girls_score := 85, combined_score := 80 }

def combined_boys_score : ℝ := 74

theorem girls_combined_average (c d : ℝ) 
  (hc : c > 0) (hd : d > 0)
  (h_central : c * central.boys_score + c * central.girls_score = (c + c) * central.combined_score)
  (h_delta : d * delta.boys_score + d * delta.girls_score = (d + d) * delta.combined_score)
  (h_boys : (c * central.boys_score + d * delta.boys_score) / (c + d) = combined_boys_score) :
  (c * central.girls_score + d * delta.girls_score) / (c + d) = 79 := by
  sorry

end NUMINAMATH_CALUDE_girls_combined_average_l2357_235726


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2357_235797

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- Slope-intercept form of L1
  (L2 P.1 P.2) →                          -- L2 passes through P
  ((-2) * (1/2) = -1) →                   -- Slopes are negative reciprocals
  ∀ x y, L1 x y → L2 x y → (x - P.1) * (x - P.1) + (y - P.2) * (y - P.2) ≠ 0 →
    (x - P.1) * (x - 2) + (y - P.2) * (y - (-3)) = 0 -- Perpendicular condition
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2357_235797


namespace NUMINAMATH_CALUDE_car_arrives_earlier_l2357_235717

/-- Represents a vehicle (car or bus) -/
inductive Vehicle
| Car
| Bus

/-- Represents the state of a traffic light -/
inductive LightState
| Green
| Red

/-- Calculates the travel time for a vehicle given the number of blocks -/
def travelTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  match v with
  | Vehicle.Car => blocks
  | Vehicle.Bus => 2 * blocks

/-- Calculates the number of complete light cycles for a given time -/
def completeLightCycles (time : ℕ) : ℕ :=
  time / 4

/-- Calculates the waiting time at red lights for a given travel time -/
def waitingTime (time : ℕ) : ℕ :=
  completeLightCycles time

/-- Calculates the total time to reach the destination for a vehicle -/
def totalTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  let travel := travelTime v blocks
  travel + waitingTime travel

/-- The main theorem to prove -/
theorem car_arrives_earlier (blocks : ℕ) (h : blocks = 12) :
  totalTime Vehicle.Car blocks + 9 = totalTime Vehicle.Bus blocks :=
by sorry

end NUMINAMATH_CALUDE_car_arrives_earlier_l2357_235717


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l2357_235793

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l2357_235793


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2357_235734

theorem polynomial_factorization (x : ℝ) : 
  45 * x^6 - 270 * x^12 + 90 * x^7 = 45 * x^6 * (1 + 2*x - 6*x^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2357_235734


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2357_235757

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-1, 0, 1}
  let B : Set ℝ := {a - 1, a + 1/a}
  (A ∩ B = {0}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2357_235757


namespace NUMINAMATH_CALUDE_sine_shift_left_specific_sine_shift_shift_result_l2357_235789

/-- Shifting a sine function to the left -/
theorem sine_shift_left (A : ℝ) (ω : ℝ) (φ : ℝ) (h : ℝ) :
  (fun x => A * Real.sin (ω * (x + h) + φ)) =
  (fun x => A * Real.sin (ω * x + (ω * h + φ))) :=
by sorry

/-- The specific case of shifting y = 3sin(2x + π/6) left by π/6 -/
theorem specific_sine_shift :
  (fun x => 3 * Real.sin (2 * x + π/6)) =
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) :=
by sorry

/-- The result of the shift is y = 3sin(2x - π/6) -/
theorem shift_result :
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) =
  (fun x => 3 * Real.sin (2 * x - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_specific_sine_shift_shift_result_l2357_235789


namespace NUMINAMATH_CALUDE_intersection_A_B_l2357_235700

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}

-- Define set B
def B : Set ℝ := {x | x - 1 > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2357_235700


namespace NUMINAMATH_CALUDE_inequality_solution_l2357_235732

theorem inequality_solution (x : ℝ) : 
  (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
  (x > -5 ∧ x < -1) ∨ (x > 4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2357_235732


namespace NUMINAMATH_CALUDE_gym_distance_difference_l2357_235752

/-- The distance from Anthony's apartment to work in miles -/
def distance_to_work : ℝ := 10

/-- The distance from Anthony's apartment to the gym in miles -/
def distance_to_gym : ℝ := 7

/-- The distance to the gym is more than half the distance to work -/
axiom gym_further_than_half : distance_to_gym > distance_to_work / 2

theorem gym_distance_difference : distance_to_gym - distance_to_work / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_distance_difference_l2357_235752


namespace NUMINAMATH_CALUDE_lunchroom_tables_l2357_235736

theorem lunchroom_tables (students_per_table : ℕ) (total_students : ℕ) (h1 : students_per_table = 6) (h2 : total_students = 204) :
  total_students / students_per_table = 34 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_tables_l2357_235736


namespace NUMINAMATH_CALUDE_travis_cereal_consumption_l2357_235740

/-- Represents the number of boxes of cereal Travis eats per week -/
def boxes_per_week : ℕ := sorry

/-- The cost of one box of cereal in dollars -/
def cost_per_box : ℚ := 3

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Travis spends on cereal in a year in dollars -/
def total_spent : ℚ := 312

theorem travis_cereal_consumption :
  boxes_per_week = 2 ∧
  cost_per_box * boxes_per_week * weeks_in_year = total_spent :=
by sorry

end NUMINAMATH_CALUDE_travis_cereal_consumption_l2357_235740


namespace NUMINAMATH_CALUDE_range_of_a_l2357_235784

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 3 * (a - 3) * x^2 + 1 / x = 0) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → 
  a ∈ Set.Iic 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2357_235784
