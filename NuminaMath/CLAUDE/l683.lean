import Mathlib

namespace intersection_of_perpendicular_tangents_on_parabola_l683_68355

/-- Given two points on the parabola y = 2x^2 with perpendicular tangents, 
    their intersection point has y-coordinate -1/2 -/
theorem intersection_of_perpendicular_tangents_on_parabola 
  (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2 * a^2)
  let B : ℝ × ℝ := (b, 2 * b^2)
  let tangent_A (x : ℝ) := 4 * a * x - 2 * a^2
  let tangent_B (x : ℝ) := 4 * b * x - 2 * b^2
  -- Condition: A and B are on the parabola y = 2x^2
  -- Condition: Tangents at A and B are perpendicular
  4 * a * 4 * b = -1 →
  -- Conclusion: The y-coordinate of the intersection point P is -1/2
  ∃ x, tangent_A x = tangent_B x ∧ tangent_A x = -1/2 :=
by sorry

end intersection_of_perpendicular_tangents_on_parabola_l683_68355


namespace factor_72x3_minus_252x7_l683_68300

theorem factor_72x3_minus_252x7 (x : ℝ) : 72 * x^3 - 252 * x^7 = 36 * x^3 * (2 - 7 * x^4) := by
  sorry

end factor_72x3_minus_252x7_l683_68300


namespace bricks_for_room_floor_bricks_needed_is_340_l683_68381

/-- Calculates the number of bricks needed for a rectangular room floor -/
theorem bricks_for_room_floor 
  (length : ℝ) 
  (breadth : ℝ) 
  (bricks_per_sqm : ℕ) 
  (h1 : length = 4) 
  (h2 : breadth = 5) 
  (h3 : bricks_per_sqm = 17) : 
  ℕ := by
  
  sorry

#check bricks_for_room_floor

/-- Proves that 340 bricks are needed for the given room dimensions -/
theorem bricks_needed_is_340 : 
  bricks_for_room_floor 4 5 17 rfl rfl rfl = 340 := by
  
  sorry

end bricks_for_room_floor_bricks_needed_is_340_l683_68381


namespace tank_capacity_proof_l683_68336

def tank_capacity (initial_loss_rate : ℕ) (initial_loss_hours : ℕ) 
  (secondary_loss_rate : ℕ) (secondary_loss_hours : ℕ)
  (fill_rate : ℕ) (fill_hours : ℕ) (remaining_to_fill : ℕ) : ℕ :=
  (initial_loss_rate * initial_loss_hours) + 
  (secondary_loss_rate * secondary_loss_hours) + 
  (fill_rate * fill_hours) + 
  remaining_to_fill

theorem tank_capacity_proof : 
  tank_capacity 32000 5 10000 10 40000 3 140000 = 520000 :=
by sorry

end tank_capacity_proof_l683_68336


namespace magazines_to_boxes_l683_68386

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end magazines_to_boxes_l683_68386


namespace two_digit_number_sum_l683_68319

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 5 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end two_digit_number_sum_l683_68319


namespace dave_toy_tickets_l683_68325

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys (initial_tickets clothes_tickets toy_extra : ℕ) : ℕ :=
  clothes_tickets + toy_extra

/-- Proof that Dave used 12 tickets to buy toys -/
theorem dave_toy_tickets :
  let initial_tickets : ℕ := 19
  let clothes_tickets : ℕ := 7
  let toy_extra : ℕ := 5
  tickets_for_toys initial_tickets clothes_tickets toy_extra = 12 := by
  sorry

end dave_toy_tickets_l683_68325


namespace imaginary_part_of_complex_fraction_l683_68361

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l683_68361


namespace galia_number_transformation_l683_68337

theorem galia_number_transformation (k : ℝ) :
  (∃ N : ℝ, ((k * N + N) / N - N = k - 100)) → (∃ N : ℝ, N = 101) :=
by sorry

end galia_number_transformation_l683_68337


namespace no_simultaneous_squares_l683_68349

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), 
  (x^2 + y = a^2) ∧ (y^2 + x = b^2) := by
  sorry

end no_simultaneous_squares_l683_68349


namespace set_problem_l683_68379

def U : Set ℕ := {x | x ≤ 20 ∧ Nat.Prime x}

theorem set_problem (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : (U \ A) ∩ B = {7, 19})
  (h3 : U \ (A ∪ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 19, 11, 13} := by
  sorry

end set_problem_l683_68379


namespace expression_evaluation_l683_68368

theorem expression_evaluation : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end expression_evaluation_l683_68368


namespace chickens_bought_l683_68365

def eggCount : ℕ := 20
def eggPrice : ℕ := 2
def chickenPrice : ℕ := 8
def totalSpent : ℕ := 88

theorem chickens_bought :
  (totalSpent - eggCount * eggPrice) / chickenPrice = 6 := by sorry

end chickens_bought_l683_68365


namespace odd_then_even_probability_l683_68335

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- Represents the bag of balls -/
def Bag : Finset Ball := sorry

/-- The bag contains 5 balls numbered 1 to 5 -/
axiom bag_content : Bag.card = 5 ∧ ∀ n : Nat, 1 ≤ n ∧ n ≤ 5 → ∃! b : Ball, b ∈ Bag ∧ b.number = n

/-- A ball is odd-numbered if its number is odd -/
def is_odd (b : Ball) : Prop := b.number % 2 = 1

/-- A ball is even-numbered if its number is even -/
def is_even (b : Ball) : Prop := b.number % 2 = 0

/-- The probability of drawing an odd-numbered ball first and an even-numbered ball second -/
def prob_odd_then_even : ℚ := sorry

/-- The main theorem to prove -/
theorem odd_then_even_probability : prob_odd_then_even = 3 / 10 := sorry

end odd_then_even_probability_l683_68335


namespace product_increase_factor_l683_68371

theorem product_increase_factor (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, (10 * a) * b = 10 * (a * b)) :=
sorry

end product_increase_factor_l683_68371


namespace brian_final_cards_l683_68387

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

theorem brian_final_cards : 
  initial_cards - cards_taken + packs_bought * cards_per_pack = 62 := by
  sorry

end brian_final_cards_l683_68387


namespace rectangular_field_perimeter_l683_68308

/-- Calculates the perimeter of a rectangular field enclosed by evenly spaced posts -/
def field_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let short_side_posts := num_posts / 3
  let short_side_gaps := short_side_posts - 1
  let short_side_length := short_side_gaps * gap_width + short_side_posts * (post_width / 12)
  let long_side_length := 2 * short_side_length
  2 * (short_side_length + long_side_length)

theorem rectangular_field_perimeter :
  field_perimeter 36 (4 / 1) (7 / 2) = 238 :=
by sorry

end rectangular_field_perimeter_l683_68308


namespace smallest_n_congruence_l683_68328

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 : ℤ)^(k : ℕ) % 5 ≠ (k : ℤ)^7 % 5) ∧ 
  (7 : ℤ)^(n : ℕ) % 5 = (n : ℤ)^7 % 5 → 
  n = 7 := by sorry

end smallest_n_congruence_l683_68328


namespace add_12345_seconds_to_5_15_00_l683_68391

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_15_00 :
  addSeconds (Time.mk 5 15 0) 12345 = Time.mk 9 0 45 := by
  sorry

end add_12345_seconds_to_5_15_00_l683_68391


namespace gcd_16016_20020_l683_68356

theorem gcd_16016_20020 : Nat.gcd 16016 20020 = 4004 := by
  sorry

end gcd_16016_20020_l683_68356


namespace julia_watch_collection_l683_68330

theorem julia_watch_collection :
  let silver_watches : ℕ := 20
  let bronze_watches : ℕ := 3 * silver_watches
  let platinum_watches : ℕ := 2 * bronze_watches
  let gold_watches : ℕ := (silver_watches + platinum_watches) / 5
  let total_watches : ℕ := silver_watches + bronze_watches + platinum_watches + gold_watches
  total_watches = 228 :=
by sorry

end julia_watch_collection_l683_68330


namespace range_of_a_l683_68367

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

-- Define set A
def A (a : ℝ) : Set ℝ := {x | f a x = x}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : A a = B a) (h2 : (A a).Nonempty) :
  a ∈ Set.Icc (-1/4 : ℝ) (3/4 : ℝ) := by
  sorry

end range_of_a_l683_68367


namespace janous_inequality_l683_68359

theorem janous_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end janous_inequality_l683_68359


namespace sum_of_square_roots_l683_68332

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) = 15 := by
  sorry

end sum_of_square_roots_l683_68332


namespace point_P_coordinates_l683_68380

def P₁ : ℝ × ℝ := (2, -1)
def P₂ : ℝ × ℝ := (0, 5)

def on_extension_line (P₁ P₂ P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (t • P₂.1 + (1 - t) • P₁.1, t • P₂.2 + (1 - t) • P₁.2)

def distance_ratio (P₁ P₂ P : ℝ × ℝ) : Prop :=
  (P.1 - P₁.1)^2 + (P.2 - P₁.2)^2 = 4 * ((P₂.1 - P.1)^2 + (P₂.2 - P.2)^2)

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, on_extension_line P₁ P₂ P → distance_ratio P₁ P₂ P → P = (-2, 11) :=
by sorry

end point_P_coordinates_l683_68380


namespace magazine_purchase_methods_l683_68384

theorem magazine_purchase_methods (n : ℕ) (m : ℕ) (total : ℕ) : 
  n + m = 11 → 
  n = 8 → 
  m = 3 → 
  total = 10 →
  (Nat.choose n 5 + Nat.choose n 4 * Nat.choose m 2) = 266 := by
  sorry

end magazine_purchase_methods_l683_68384


namespace store_inventory_theorem_l683_68375

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy headphones, keyboard, and mouse --/
def ways_to_buy (inv : Inventory) : ℕ :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy the items --/
theorem store_inventory_theorem (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy inv = 646 := by
  sorry

#eval ways_to_buy { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end store_inventory_theorem_l683_68375


namespace bounded_recurrence_periodic_l683_68358

def is_bounded (x : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n, |x n| ≤ M

def recurrence_relation (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 5) = (5 * (x (n + 4))^3 + x (n + 3) - 3 * x (n + 2) + x n) /
                   (2 * x (n + 2) + (x (n + 1))^2 + x (n + 1) * x n)

def eventually_periodic (x : ℕ → ℤ) : Prop :=
  ∃ k p : ℕ, p > 0 ∧ ∀ n ≥ k, x (n + p) = x n

theorem bounded_recurrence_periodic
  (x : ℕ → ℤ) (h_bounded : is_bounded x) (h_recurrence : recurrence_relation x) :
  eventually_periodic x :=
sorry

end bounded_recurrence_periodic_l683_68358


namespace snail_square_exists_l683_68329

/-- A natural number is a "snail" number if it can be formed by concatenating
    three consecutive natural numbers in some order. -/
def is_snail (n : ℕ) : Prop :=
  ∃ a b c : ℕ, b = a + 1 ∧ c = b + 1 ∧
  (n.repr = a.repr ++ b.repr ++ c.repr ∨
   n.repr = a.repr ++ c.repr ++ b.repr ∨
   n.repr = b.repr ++ a.repr ++ c.repr ∨
   n.repr = b.repr ++ c.repr ++ a.repr ∨
   n.repr = c.repr ++ a.repr ++ b.repr ∨
   n.repr = c.repr ++ b.repr ++ a.repr)

theorem snail_square_exists :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_snail n ∧ ∃ m : ℕ, n = m^2 :=
by
  use 1089
  sorry

end snail_square_exists_l683_68329


namespace half_oz_mixture_bubbles_l683_68301

/-- The number of bubbles that can be made from one ounce of Dawn liquid soap -/
def dawn_bubbles_per_oz : ℕ := 200000

/-- The number of bubbles that can be made from one ounce of Dr. Bronner's liquid soap -/
def bronner_bubbles_per_oz : ℕ := 2 * dawn_bubbles_per_oz

/-- The number of bubbles that can be made from one ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps -/
def mixture_bubbles_per_oz : ℕ := (dawn_bubbles_per_oz + bronner_bubbles_per_oz) / 2

/-- Theorem: One half ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps can make 150,000 bubbles -/
theorem half_oz_mixture_bubbles : mixture_bubbles_per_oz / 2 = 150000 := by
  sorry

end half_oz_mixture_bubbles_l683_68301


namespace alice_bob_sum_l683_68385

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is a perfect square if it's the product of an integer with itself. -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_sum : 
  ∀ (A B : ℕ),
  (A ≠ 1) →  -- Alice's number is not the smallest
  (B = 2) →  -- Bob's number is the smallest prime
  (isPrime B) →
  (isPerfectSquare (100 * B + A)) →
  (1 ≤ A ∧ A ≤ 40) →  -- Alice's number is between 1 and 40
  (1 ≤ B ∧ B ≤ 40) →  -- Bob's number is between 1 and 40
  (A + B = 27) := by sorry

end alice_bob_sum_l683_68385


namespace shortened_area_l683_68390

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

/-- Theorem stating the relationship between the original rectangle and the shortened rectangle --/
theorem shortened_area (h : area shortened = 21) :
  ∃ (r : Rectangle), r.length = original.length ∧ r.width = original.width - 2 ∧ area r = 25 := by
  sorry


end shortened_area_l683_68390


namespace number_of_students_l683_68341

theorem number_of_students : ∃ (x : ℕ), 
  (∃ (total : ℕ), total = 3 * x + 8) ∧ 
  (5 * (x - 1) + 3 > 3 * x + 8) ∧
  (3 * x + 8 ≥ 5 * (x - 1)) ∧
  x = 6 := by
  sorry

end number_of_students_l683_68341


namespace max_value_of_a_l683_68378

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_one : a^2 + b^2 + c^2 = 1) :
  a ≤ Real.sqrt 6 / 3 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧ a₀ = Real.sqrt 6 / 3 :=
by sorry

end max_value_of_a_l683_68378


namespace tan_four_thirds_pi_l683_68324

theorem tan_four_thirds_pi : Real.tan (4 * π / 3) = Real.sqrt 3 := by
  sorry

end tan_four_thirds_pi_l683_68324


namespace jacket_trouser_combinations_l683_68360

theorem jacket_trouser_combinations (jacket_styles : ℕ) (trouser_colors : ℕ) : 
  jacket_styles = 4 → trouser_colors = 3 → jacket_styles * trouser_colors = 12 := by
  sorry

end jacket_trouser_combinations_l683_68360


namespace shooter_probability_l683_68322

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end shooter_probability_l683_68322


namespace work_completion_time_l683_68334

theorem work_completion_time (b_time : ℕ) (joint_time : ℕ) (b_remaining_time : ℕ) :
  b_time = 40 → joint_time = 9 → b_remaining_time = 23 →
  ∃ (a_time : ℕ),
    (joint_time : ℚ) * ((1 : ℚ) / a_time + (1 : ℚ) / b_time) + 
    (b_remaining_time : ℚ) * ((1 : ℚ) / b_time) = 1 ∧
    a_time = 45 :=
by sorry

end work_completion_time_l683_68334


namespace jenna_round_trip_pay_l683_68357

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (rate : ℚ) (one_way_distance : ℚ) : ℚ :=
  2 * rate * one_way_distance

/-- Proves that the round trip pay for a rate of $0.40 per mile and 400 miles one-way is $320 -/
theorem jenna_round_trip_pay :
  round_trip_pay (40 / 100) 400 = 320 := by
  sorry

#eval round_trip_pay (40 / 100) 400

end jenna_round_trip_pay_l683_68357


namespace ratio_of_numbers_l683_68302

theorem ratio_of_numbers (a b : ℕ) (h1 : a > b) (h2 : a + b = 96) (h3 : a = 64) (h4 : b = 32) : a / b = 2 := by
  sorry

end ratio_of_numbers_l683_68302


namespace third_guinea_pig_eats_more_l683_68326

/-- The number of guinea pigs Rollo has -/
def num_guinea_pigs : ℕ := 3

/-- The amount of food the first guinea pig eats in cups -/
def first_guinea_pig_food : ℕ := 2

/-- The amount of food the second guinea pig eats in cups -/
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food

/-- The total amount of food for all guinea pigs in cups -/
def total_food : ℕ := 13

/-- The amount of food the third guinea pig eats in cups -/
def third_guinea_pig_food : ℕ := total_food - first_guinea_pig_food - second_guinea_pig_food

theorem third_guinea_pig_eats_more :
  third_guinea_pig_food = second_guinea_pig_food + 3 := by
  sorry

end third_guinea_pig_eats_more_l683_68326


namespace function_inequality_condition_l683_68346

theorem function_inequality_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x : ℝ, |x + 0.4| < b → |5 * x - 3 + 1| < a) ↔ b ≤ a / 5 := by
  sorry

end function_inequality_condition_l683_68346


namespace total_earnings_equals_9780_l683_68377

-- Define the earnings of each individual
def salvadore_earnings : ℕ := 1956

-- Santo's earnings are half of Salvadore's
def santo_earnings : ℕ := salvadore_earnings / 2

-- Maria's earnings are three times Santo's
def maria_earnings : ℕ := santo_earnings * 3

-- Pedro's earnings are the sum of Santo's and Maria's
def pedro_earnings : ℕ := santo_earnings + maria_earnings

-- Total earnings of all four individuals
def total_earnings : ℕ := salvadore_earnings + santo_earnings + maria_earnings + pedro_earnings

-- Theorem statement
theorem total_earnings_equals_9780 : total_earnings = 9780 := by
  sorry

end total_earnings_equals_9780_l683_68377


namespace y1_greater_y2_l683_68395

/-- A linear function passing through the first, second, and fourth quadrants -/
structure QuadrantCrossingLine where
  m : ℝ
  n : ℝ
  first_quadrant : ∃ x > 0, m * x + n > 0
  second_quadrant : ∃ x < 0, m * x + n > 0
  fourth_quadrant : ∃ x > 0, m * x + n < 0

/-- Theorem: For a linear function y = mx + n passing through the first, second, and fourth quadrants,
    if (1, y₁) and (3, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_y2 (line : QuadrantCrossingLine) (y₁ y₂ : ℝ)
    (point1 : line.m * 1 + line.n = y₁)
    (point2 : line.m * 3 + line.n = y₂) :
    y₁ > y₂ := by
  sorry

end y1_greater_y2_l683_68395


namespace impossible_to_use_all_stock_l683_68364

/-- Represents the number of units required for each product type -/
structure ProductRequirements where
  alpha_A : Nat
  alpha_B : Nat
  beta_B : Nat
  beta_C : Nat
  gamma_A : Nat
  gamma_C : Nat

/-- Represents the current stock levels after production -/
structure StockLevels where
  remaining_A : Nat
  remaining_B : Nat
  remaining_C : Nat

/-- Theorem stating the impossibility of using up all stocks exactly -/
theorem impossible_to_use_all_stock 
  (req : ProductRequirements)
  (stock : StockLevels)
  (h_req : req = { 
    alpha_A := 2, alpha_B := 2, 
    beta_B := 1, beta_C := 1, 
    gamma_A := 2, gamma_C := 1 
  })
  (h_stock : stock = { remaining_A := 2, remaining_B := 1, remaining_C := 0 }) :
  ∀ (p q r : Nat), ∃ (total_A total_B total_C : Nat),
    (2 * p + 2 * r + stock.remaining_A ≠ total_A) ∨
    (2 * p + q + stock.remaining_B ≠ total_B) ∨
    (q + r ≠ total_C) :=
sorry

end impossible_to_use_all_stock_l683_68364


namespace units_digit_of_7_to_103_l683_68340

theorem units_digit_of_7_to_103 : ∃ n : ℕ, 7^103 ≡ 3 [ZMOD 10] :=
by
  sorry

end units_digit_of_7_to_103_l683_68340


namespace inequality_solution_range_l683_68362

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end inequality_solution_range_l683_68362


namespace ruined_tomatoes_percentage_l683_68317

/-- The percentage of ruined and discarded tomatoes -/
def ruined_percentage : ℝ := 15

/-- The purchase price per pound of tomatoes -/
def purchase_price : ℝ := 0.80

/-- The desired profit percentage on the cost of tomatoes -/
def profit_percentage : ℝ := 8

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 1.0165

/-- Theorem stating that given the purchase price, profit percentage, and selling price,
    the percentage of ruined and discarded tomatoes is approximately 15% -/
theorem ruined_tomatoes_percentage :
  ∀ (W : ℝ), W > 0 →
  selling_price * (100 - ruined_percentage) / 100 * W - purchase_price * W =
  profit_percentage / 100 * purchase_price * W :=
by sorry

end ruined_tomatoes_percentage_l683_68317


namespace negative_reciprocal_inequality_l683_68351

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -1/a < -1/b := by
  sorry

end negative_reciprocal_inequality_l683_68351


namespace line_vector_to_slope_intercept_l683_68369

/-- Given a line equation in vector form, prove its slope-intercept form -/
theorem line_vector_to_slope_intercept 
  (x y : ℝ) : 
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 5) = 0 ↔ y = 2 * x - 13 := by
  sorry

end line_vector_to_slope_intercept_l683_68369


namespace word_permutation_ratio_l683_68383

theorem word_permutation_ratio : 
  let n₁ : ℕ := 6  -- number of letters in "СКАЛКА"
  let n₂ : ℕ := 7  -- number of letters in "ТЕФТЕЛЬ"
  let r : ℕ := 2   -- number of repeated letters in each word
  
  -- number of distinct permutations for each word
  let perm₁ : ℕ := n₁! / (r! * r!)
  let perm₂ : ℕ := n₂! / (r! * r!)

  perm₂ / perm₁ = 7 := by
  sorry

end word_permutation_ratio_l683_68383


namespace natural_number_pairs_l683_68314

theorem natural_number_pairs : ∀ a b : ℕ,
  (90 < a + b ∧ a + b < 100) ∧ (9/10 < (a : ℚ) / (b : ℚ) ∧ (a : ℚ) / (b : ℚ) < 91/100) ↔
  ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) := by
  sorry

end natural_number_pairs_l683_68314


namespace proportion_solution_l683_68318

theorem proportion_solution (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 := by
  sorry

end proportion_solution_l683_68318


namespace min_value_of_f_l683_68306

/-- The function f(x) = 3x^2 - 6x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 9

/-- The minimum value of f(x) is 6 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 6 := by
  sorry

end min_value_of_f_l683_68306


namespace correct_managers_in_sample_l683_68393

/-- Calculates the number of managers to be drawn in a stratified sample -/
def managers_in_sample (total_employees : ℕ) (total_managers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_managers * sample_size) / total_employees

theorem correct_managers_in_sample :
  managers_in_sample 160 32 20 = 4 := by
  sorry

end correct_managers_in_sample_l683_68393


namespace max_value_x_plus_7y_l683_68345

theorem max_value_x_plus_7y :
  ∀ x y : ℝ,
  0 ≤ x ∧ x ≤ 1 →
  0 ≤ y ∧ y ≤ 1 →
  Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + Real.sqrt (y * (1 - x)) / Real.sqrt 7 →
  (∀ z w : ℝ, 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → z + 7 * w ≤ 57 / 8) ∧
  ∃ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ a + 7 * b = 57 / 8 :=
by sorry

end max_value_x_plus_7y_l683_68345


namespace sarahs_age_proof_l683_68354

theorem sarahs_age_proof (ana billy mark sarah : ℕ) : 
  ana + 8 = 40 → 
  billy = ana / 2 → 
  mark = billy + 4 → 
  sarah = 3 * mark - 4 → 
  sarah = 56 := by
  sorry

end sarahs_age_proof_l683_68354


namespace quadratic_roots_range_l683_68372

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (k-2)*x + 2*k-1 = 0 ↔ x = x₁ ∨ x = x₂) →
  0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
  1/2 < k ∧ k < 2/3 :=
sorry

end quadratic_roots_range_l683_68372


namespace exponential_plus_x_increasing_l683_68382

open Real

theorem exponential_plus_x_increasing (x : ℝ) : exp (x + 1) + (x + 1) > (exp x + x) + 1 := by
  sorry

end exponential_plus_x_increasing_l683_68382


namespace cody_tickets_l683_68394

/-- The number of tickets Cody spent on a beanie -/
def beanie_cost : ℕ := 25

/-- The number of additional tickets Cody won later -/
def additional_tickets : ℕ := 6

/-- The number of tickets Cody has now -/
def current_tickets : ℕ := 30

/-- The initial number of tickets Cody won -/
def initial_tickets : ℕ := 49

theorem cody_tickets : 
  initial_tickets = beanie_cost + (current_tickets - additional_tickets) :=
by sorry

end cody_tickets_l683_68394


namespace a_range_for_unique_positive_zero_l683_68316

/-- The function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The statement that f has only one zero point -/
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem a_range_for_unique_positive_zero (a : ℝ) :
  (has_unique_zero (f a)) ∧ (∃ x₀ > 0, f a x₀ = 0) → a < -2 :=
sorry

end a_range_for_unique_positive_zero_l683_68316


namespace joyce_apples_l683_68307

/-- The number of apples Joyce ends up with after giving some away -/
def apples_remaining (starting_apples given_away : ℕ) : ℕ :=
  starting_apples - given_away

/-- Theorem stating that Joyce ends up with 23 apples -/
theorem joyce_apples : apples_remaining 75 52 = 23 := by
  sorry

end joyce_apples_l683_68307


namespace factor_tree_X_value_l683_68343

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    Y = 4 * F →
    F = 2 * F →
    Z = 7 * G →
    G = 7 * G →
    X = Y * Z →
    X = 392 := by
  sorry

end factor_tree_X_value_l683_68343


namespace interval_length_implies_difference_l683_68398

/-- Given an inequality a ≤ 3x + 6 ≤ b, if the length of the interval of solutions is 15, then b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ (l : ℝ), l = 15 ∧ l = (b - 6) / 3 - (a - 6) / 3) → b - a = 45 := by
  sorry

end interval_length_implies_difference_l683_68398


namespace line_in_quadrants_implies_positive_slope_l683_68399

/-- A line passing through the first and third quadrants -/
structure LineInQuadrants where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x

/-- If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_quadrants_implies_positive_slope (l : LineInQuadrants) : l.k > 0 := by
  sorry

end line_in_quadrants_implies_positive_slope_l683_68399


namespace unique_M_condition_l683_68311

theorem unique_M_condition (M : ℝ) : 
  (M > 0 ∧ 
   (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
     (a + M / (a * b) ≥ 1 + M ∨ 
      b + M / (b * c) ≥ 1 + M ∨ 
      c + M / (c * a) ≥ 1 + M))) ↔ 
  M = 1 / 2 := by sorry

end unique_M_condition_l683_68311


namespace cosine_function_vertical_shift_l683_68305

/-- Given a cosine function y = a * cos(b * x + c) + d that oscillates between 5 and -3,
    prove that the vertical shift d equals 1. -/
theorem cosine_function_vertical_shift
  (a b c d : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_oscillation : ∀ x, -3 ≤ a * Real.cos (b * x + c) + d ∧ 
                        a * Real.cos (b * x + c) + d ≤ 5) :
  d = 1 := by
sorry

end cosine_function_vertical_shift_l683_68305


namespace sum_remainder_mod_11_l683_68376

theorem sum_remainder_mod_11 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end sum_remainder_mod_11_l683_68376


namespace bryan_mineral_samples_l683_68320

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The number of shelves -/
def number_of_shelves : ℕ := 7

/-- The total number of mineral samples -/
def total_samples : ℕ := samples_per_shelf * number_of_shelves

theorem bryan_mineral_samples :
  total_samples = 455 :=
sorry

end bryan_mineral_samples_l683_68320


namespace triangle_angle_inequality_l683_68353

theorem triangle_angle_inequality (X Y Z : ℝ) 
  (h_positive : X > 0 ∧ Y > 0 ∧ Z > 0) 
  (h_sum : 2 * X + 2 * Y + 2 * Z = π) : 
  (Real.sin X / Real.cos (Y - Z)) + 
  (Real.sin Y / Real.cos (Z - X)) + 
  (Real.sin Z / Real.cos (X - Y)) ≥ 3 / 2 := by
  sorry

end triangle_angle_inequality_l683_68353


namespace roots_of_quadratic_equation_l683_68323

theorem roots_of_quadratic_equation :
  let f : ℂ → ℂ := λ x => x^2 + 4
  ∀ x : ℂ, f x = 0 ↔ x = 2*I ∨ x = -2*I :=
by sorry

end roots_of_quadratic_equation_l683_68323


namespace water_depth_of_specific_tower_l683_68363

/-- Represents a conical tower -/
structure ConicalTower where
  height : ℝ
  volumeAboveWater : ℝ

/-- Calculates the depth of water at the base of a conical tower -/
def waterDepth (tower : ConicalTower) : ℝ :=
  tower.height * (1 - (tower.volumeAboveWater)^(1/3))

/-- The theorem stating the depth of water for a specific conical tower -/
theorem water_depth_of_specific_tower :
  let tower : ConicalTower := ⟨10000, 1/4⟩
  waterDepth tower = 905 := by sorry

end water_depth_of_specific_tower_l683_68363


namespace parallelogram_diagonal_sum_l683_68321

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_property : (C - B) = (D - A) ∧ (D - C) = (B - A))

-- Theorem statement
theorem parallelogram_diagonal_sum 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.B - ABCD.A + (ABCD.D - ABCD.A) = ABCD.C - ABCD.A :=
sorry

end parallelogram_diagonal_sum_l683_68321


namespace max_diff_inequality_l683_68327

open Function Set

variable {n : ℕ}

/-- Two strictly increasing finite sequences of real numbers -/
def StrictlyIncreasingSeq (a b : Fin n → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j ∧ b i < b j

theorem max_diff_inequality
  (a b : Fin n → ℝ)
  (h_inc : StrictlyIncreasingSeq a b)
  (f : Fin n → Fin n)
  (h_bij : Bijective f) :
  (⨆ i, |a i - b i|) ≤ (⨆ i, |a i - b (f i)|) :=
sorry

end max_diff_inequality_l683_68327


namespace last_digit_periodicity_last_digits_first_five_l683_68313

def a (n : ℕ+) : ℕ := (n - 1) * n

theorem last_digit_periodicity (n : ℕ+) :
  ∃ (k : ℕ+), a (n + 5 * k) % 10 = a n % 10 :=
sorry

theorem last_digits_first_five :
  (a 1 % 10 = 0) ∧
  (a 2 % 10 = 2) ∧
  (a 3 % 10 = 6) ∧
  (a 4 % 10 = 2) ∧
  (a 5 % 10 = 0) :=
sorry

end last_digit_periodicity_last_digits_first_five_l683_68313


namespace victors_friend_wins_checkers_game_wins_l683_68370

theorem victors_friend_wins (victor_wins : ℕ) (ratio_victor : ℕ) (ratio_friend : ℕ) : ℕ :=
  let friend_wins := (victor_wins * ratio_friend) / ratio_victor
  friend_wins

theorem checkers_game_wins : victors_friend_wins 36 9 5 = 20 := by
  sorry

end victors_friend_wins_checkers_game_wins_l683_68370


namespace correct_age_ranking_l683_68388

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define the age relation
def OlderThan : Friend → Friend → Prop := sorry

-- Define the statements
def Statement1 : Prop := ∀ f : Friend, f ≠ Friend.Emma → OlderThan Friend.Emma f
def Statement2 : Prop := ∃ f : Friend, OlderThan f Friend.Fiona
def Statement3 : Prop := ∃ f : Friend, OlderThan Friend.David f
def Statement4 : Prop := ∃ f : Friend, OlderThan f Friend.George

-- Define the theorem
theorem correct_age_ranking :
  (∀ f1 f2 : Friend, f1 ≠ f2 → (OlderThan f1 f2 ∨ OlderThan f2 f1)) →
  (Statement1 ∨ Statement2 ∨ Statement3 ∨ Statement4) →
  (¬Statement1 ∨ ¬Statement2 ∨ ¬Statement3 ∨ ¬Statement4) →
  (OlderThan Friend.Fiona Friend.Emma ∧
   OlderThan Friend.Emma Friend.George ∧
   OlderThan Friend.George Friend.David) :=
by sorry

end correct_age_ranking_l683_68388


namespace unique_solution_cube_difference_square_l683_68312

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ+,
  y.val.Prime → 
  (¬ 3 ∣ z.val) → 
  (¬ y.val ∣ z.val) → 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 := by
sorry

end unique_solution_cube_difference_square_l683_68312


namespace range_of_m_l683_68309

theorem range_of_m (x y m : ℝ) 
  (hx : 1 < x ∧ x < 3) 
  (hy : -3 < y ∧ y < 1) 
  (hm : m = x - 3*y) : 
  -2 < m ∧ m < 12 := by
sorry

end range_of_m_l683_68309


namespace geometric_sequence_ratio_l683_68344

/-- Given a geometric sequence {a_n} where each term is positive and 2a_1 + a_2 = a_3,
    prove that (a_4 + a_5) / (a_3 + a_4) = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end geometric_sequence_ratio_l683_68344


namespace simplification_and_exponent_sum_l683_68392

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^3 * z^8) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z * (5 * x^2 * z^5) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 1 = 3) := by sorry

end simplification_and_exponent_sum_l683_68392


namespace club_simplifier_probability_l683_68339

def probability_more_wins_than_losses (num_matches : ℕ) 
  (prob_win prob_lose prob_tie : ℚ) : ℚ :=
  sorry

theorem club_simplifier_probability :
  probability_more_wins_than_losses 3 (1/2) (1/4) (1/4) = 25/64 :=
by sorry

end club_simplifier_probability_l683_68339


namespace square_perimeter_from_p_shape_l683_68310

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the P shape formed by rectangles -/
structure PShape where
  rectangles : Fin 4 → Rectangle

theorem square_perimeter_from_p_shape 
  (s : Square) 
  (p : PShape) 
  (h1 : ∀ i, p.rectangles i = ⟨s.side / 5, 4 * s.side / 5⟩) 
  (h2 : (6 * (4 * s.side / 5) + 4 * (s.side / 5) : ℝ) = 56) :
  4 * s.side = 40 := by
sorry

end square_perimeter_from_p_shape_l683_68310


namespace can_tile_4x7_with_4x1_l683_68396

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetromino with width and height -/
structure Tetromino where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle can be tiled with a given tetromino -/
def can_tile (r : Rectangle) (t : Tetromino) : Prop :=
  ∃ (n : ℕ), n * (t.width * t.height) = r.width * r.height

/-- The 4x7 rectangle -/
def rectangle_4x7 : Rectangle :=
  { width := 4, height := 7 }

/-- The 4x1 tetromino -/
def tetromino_4x1 : Tetromino :=
  { width := 4, height := 1 }

/-- Theorem stating that the 4x7 rectangle can be tiled with 4x1 tetrominos -/
theorem can_tile_4x7_with_4x1 : can_tile rectangle_4x7 tetromino_4x1 :=
  sorry

end can_tile_4x7_with_4x1_l683_68396


namespace customers_who_left_l683_68373

-- Define the initial number of customers
def initial_customers : ℕ := 13

-- Define the number of new customers
def new_customers : ℕ := 4

-- Define the final number of customers
def final_customers : ℕ := 9

-- Theorem to prove the number of customers who left
theorem customers_who_left :
  ∃ (left : ℕ), initial_customers - left + new_customers = final_customers ∧ left = 8 :=
by sorry

end customers_who_left_l683_68373


namespace siblings_average_age_l683_68303

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) :
  youngest_age = 17 →
  age_differences = [4, 5, 7] →
  (youngest_age + (age_differences.map (λ d => youngest_age + d)).sum) / 4 = 21 :=
by sorry

end siblings_average_age_l683_68303


namespace new_person_weight_l683_68333

theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 8 →
  replaced_weight = 35 →
  avg_increase = 2.5 →
  (initial_count * avg_increase + replaced_weight : Real) = 55 :=
by sorry

end new_person_weight_l683_68333


namespace clearance_sale_prices_l683_68350

/-- Calculates the final price after applying two successive discounts --/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2)

/-- Proves that the final prices of the hat and gloves are correct --/
theorem clearance_sale_prices 
  (hatInitialPrice : ℝ) 
  (hatDiscount1 : ℝ) 
  (hatDiscount2 : ℝ)
  (glovesInitialPrice : ℝ) 
  (glovesDiscount1 : ℝ) 
  (glovesDiscount2 : ℝ)
  (hatInitialPrice_eq : hatInitialPrice = 15)
  (hatDiscount1_eq : hatDiscount1 = 0.20)
  (hatDiscount2_eq : hatDiscount2 = 0.40)
  (glovesInitialPrice_eq : glovesInitialPrice = 8)
  (glovesDiscount1_eq : glovesDiscount1 = 0.25)
  (glovesDiscount2_eq : glovesDiscount2 = 0.30) :
  finalPrice hatInitialPrice hatDiscount1 hatDiscount2 = 7.20 ∧
  finalPrice glovesInitialPrice glovesDiscount1 glovesDiscount2 = 4.20 := by
  sorry

#check clearance_sale_prices

end clearance_sale_prices_l683_68350


namespace binomial_coefficient_equality_l683_68347

theorem binomial_coefficient_equality (n : ℕ+) : 
  (Nat.choose n.val 2 = Nat.choose n.val 3) → n = 5 := by
  sorry

end binomial_coefficient_equality_l683_68347


namespace dog_bones_problem_l683_68352

theorem dog_bones_problem (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 493)
  (h2 : final_bones = 860) :
  final_bones - initial_bones = 367 := by
  sorry

end dog_bones_problem_l683_68352


namespace consecutive_triangle_altitude_l683_68315

/-- Represents a triangle with consecutive integer side lengths -/
structure ConsecutiveTriangle where
  a : ℕ
  is_acute : a > 0

/-- The altitude from the vertex opposite the middle side -/
def altitude (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The length of the shorter segment of the middle side -/
def shorter_segment (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The length of the longer segment of the middle side -/
def longer_segment (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The theorem stating the properties of the triangle -/
theorem consecutive_triangle_altitude (t : ConsecutiveTriangle) :
  (longer_segment t - shorter_segment t = 4) →
  (∃ n : ℕ, altitude t = n) →
  t.a + 1 ≥ 26 :=
sorry

end consecutive_triangle_altitude_l683_68315


namespace sailing_speed_calculation_l683_68331

/-- The sailing speed of a ship in still water, given the following conditions:
  * Two ships (Knight and Warrior) depart from ports A and B at 8 a.m.
  * They travel towards each other, turn around at the opposite port, and return to their starting points.
  * Both ships return to their starting points at 10 a.m.
  * The time it takes for the ships to travel in the same direction is 10 minutes.
  * The speed of the current is 0.5 meters per second.
-/
def sailing_speed : ℝ := 6

/-- The speed of the current in meters per second. -/
def current_speed : ℝ := 0.5

/-- The time it takes for the ships to travel in the same direction, in seconds. -/
def same_direction_time : ℝ := 600

/-- The total travel time for each ship, in seconds. -/
def total_travel_time : ℝ := 7200

theorem sailing_speed_calculation :
  let v := sailing_speed
  let c := current_speed
  let t := same_direction_time
  let T := total_travel_time
  (v + c) * t + (v - c) * t = v * T ∧
  2 * ((v + c)⁻¹ + (v - c)⁻¹) * (v * t) = T :=
by sorry

#check sailing_speed_calculation

end sailing_speed_calculation_l683_68331


namespace integral_x_squared_sqrt_25_minus_x_squared_l683_68304

theorem integral_x_squared_sqrt_25_minus_x_squared : 
  ∫ x in (0)..(5), x^2 * Real.sqrt (25 - x^2) = (625 * Real.pi) / 16 := by
  sorry

end integral_x_squared_sqrt_25_minus_x_squared_l683_68304


namespace tangent_product_l683_68389

theorem tangent_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7)
  (h2 : 2 * Real.sin (2*x - 2*y) = Real.sin (2*x) * Real.sin (2*y)) :
  Real.tan x * Real.tan y = -7/6 := by
  sorry

end tangent_product_l683_68389


namespace largest_number_l683_68366

theorem largest_number (a b c d : ℝ) : 
  a + (b + c + d) / 3 = 92 →
  b + (a + c + d) / 3 = 86 →
  c + (a + b + d) / 3 = 80 →
  d + (a + b + c) / 3 = 90 →
  max a (max b (max c d)) = 51 := by
sorry

end largest_number_l683_68366


namespace correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l683_68338

theorem correct_sqrt_product : ∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) := by sorry

theorem incorrect_sqrt_sum : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b) := by sorry

theorem incorrect_sqrt_diff : ∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3 := by sorry

theorem incorrect_sqrt_div : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2 := by sorry

theorem only_sqrt_product_correct :
  (∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b)) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b)) ∧
  (∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2) := by sorry

end correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l683_68338


namespace count_not_divisible_1999_l683_68397

def count_not_divisible (n : ℕ) : ℕ :=
  n - (n / 4 + n / 6 - n / 12)

theorem count_not_divisible_1999 :
  count_not_divisible 1999 = 1333 := by
  sorry

end count_not_divisible_1999_l683_68397


namespace sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l683_68342

-- Define the polynomial coefficients
def a₀ : ℝ := sorry
def a₁ : ℝ := sorry
def a₂ : ℝ := sorry
def a₃ : ℝ := sorry
def a₄ : ℝ := sorry

-- Define the polynomial equation
axiom polynomial_eq : ∀ x : ℝ, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4

-- Define S
def S : ℕ := (Finset.range 27).sum (fun k => Nat.choose 27 (k + 1))

-- Theorem statements
theorem sum_of_all_coeff : a₀ + a₁ + a₂ + a₃ + a₄ = 16 := by sorry

theorem sum_of_even_coeff : a₀ + a₂ + a₄ = 136 := by sorry

theorem sum_of_coeff_except_a₀ : a₁ + a₂ + a₃ + a₄ = 15 := by sorry

theorem S_mod_9 : S % 9 = 7 := by sorry

end sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l683_68342


namespace triangle_angle_measure_l683_68374

theorem triangle_angle_measure (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sine : (7 / Real.sin A) = (8 / Real.sin B) ∧ (8 / Real.sin B) = (13 / Real.sin C)) : 
  C = (2 * π) / 3 := by
  sorry

end triangle_angle_measure_l683_68374


namespace unique_solution_lcm_gcd_equation_l683_68348

theorem unique_solution_lcm_gcd_equation :
  ∃! n : ℕ+, n.val > 0 ∧ Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 :=
by
  sorry

end unique_solution_lcm_gcd_equation_l683_68348
