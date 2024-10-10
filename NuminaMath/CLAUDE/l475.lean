import Mathlib

namespace complex_quadrant_l475_47519

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end complex_quadrant_l475_47519


namespace complex_subtraction_simplification_l475_47544

theorem complex_subtraction_simplification :
  (-5 - 3 * Complex.I) - (2 - 5 * Complex.I) = -7 + 2 * Complex.I := by
  sorry

end complex_subtraction_simplification_l475_47544


namespace merchant_salt_price_l475_47547

/-- Represents the price per pound of the unknown salt in cents -/
def unknown_price : ℝ := 50

/-- The weight of the unknown salt in pounds -/
def unknown_weight : ℝ := 20

/-- The weight of the known salt in pounds -/
def known_weight : ℝ := 40

/-- The price per pound of the known salt in cents -/
def known_price : ℝ := 35

/-- The selling price per pound of the mixture in cents -/
def selling_price : ℝ := 48

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.2

theorem merchant_salt_price :
  unknown_price = 50 ∧
  (unknown_price * unknown_weight + known_price * known_weight) * (1 + profit_percentage) =
    selling_price * (unknown_weight + known_weight) :=
by sorry

end merchant_salt_price_l475_47547


namespace sum_zero_inequality_l475_47588

theorem sum_zero_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  (a*b + a*c + a*d + b*c + b*d + c*d)^2 + 12 ≥ 6*(a*b*c + a*b*d + a*c*d + b*c*d) := by
  sorry

end sum_zero_inequality_l475_47588


namespace chess_piece_arrangements_l475_47502

theorem chess_piece_arrangements (n m : ℕ) (hn : n = 9) (hm : m = 6) :
  (Finset.card (Finset.univ : Finset (Fin n → Fin m))) = (14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6) := by
  sorry

end chess_piece_arrangements_l475_47502


namespace sphere_surface_area_l475_47562

theorem sphere_surface_area (r h : ℝ) (h1 : r = 1) (h2 : h = Real.sqrt 3) : 
  let R := (2 * Real.sqrt 3) / 3
  4 * π * R^2 = (16 * π) / 3 :=
by sorry

end sphere_surface_area_l475_47562


namespace x_squared_minus_one_necessary_not_sufficient_l475_47548

theorem x_squared_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x - 1 = 0 → x^2 - 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x - 1 = 0) :=
by sorry

end x_squared_minus_one_necessary_not_sufficient_l475_47548


namespace max_volume_rect_prism_l475_47567

/-- A right prism with rectangular bases -/
structure RectPrism where
  a : ℝ  -- length of base
  b : ℝ  -- width of base
  h : ℝ  -- height of prism
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- The sum of areas of three mutually adjacent faces is 48 -/
def adjacent_faces_area (p : RectPrism) : ℝ :=
  p.a * p.h + p.b * p.h + p.a * p.b

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.a * p.b * p.h

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_rect_prism :
  ∃ (p : RectPrism),
    adjacent_faces_area p = 48 ∧
    p.a = p.b ∧  -- two lateral faces are congruent
    ∀ (q : RectPrism),
      adjacent_faces_area q = 48 →
      q.a = q.b →
      volume q ≤ volume p ∧
      volume p = 64 :=
by sorry

end max_volume_rect_prism_l475_47567


namespace dave_new_cards_l475_47583

/-- Calculates the number of new baseball cards given the total pages used,
    cards per page, and number of old cards. -/
def new_cards (pages : ℕ) (cards_per_page : ℕ) (old_cards : ℕ) : ℕ :=
  pages * cards_per_page - old_cards

theorem dave_new_cards :
  new_cards 2 8 13 = 3 := by
  sorry

end dave_new_cards_l475_47583


namespace impossible_number_composition_l475_47584

def is_base_five_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 45

def compose_number (base_numbers : List ℕ) : ℕ := sorry

theorem impossible_number_composition :
  ¬ ∃ (x : ℕ) (base_numbers : List ℕ) (p q : ℕ),
    (base_numbers.length = 2021) ∧
    (∀ n ∈ base_numbers, is_base_five_two_digit n) ∧
    (∀ i, i < 2021 → i % 2 = 0 →
      base_numbers.get! i = base_numbers.get! (i + 1) - 1) ∧
    (x = compose_number base_numbers) ∧
    (Nat.Prime p ∧ Nat.Prime q) ∧
    (p * q = x) ∧
    (q = p + 2) :=
  sorry

end impossible_number_composition_l475_47584


namespace inequality_proof_l475_47559

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

def A : Set ℝ := {x | f x ≤ 6}

theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end inequality_proof_l475_47559


namespace gcd_of_abcd_plus_dcba_l475_47525

theorem gcd_of_abcd_plus_dcba :
  ∃ (g : ℕ), g > 1 ∧ 
  (∀ (a : ℕ), 0 ≤ a → a ≤ 3 → 
    g ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
        (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a)) ∧
  (∀ (d : ℕ), d > g → 
    ∃ (a : ℕ), 0 ≤ a ∧ a ≤ 3 ∧ 
      ¬(d ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
           (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a))) ∧
  g = 2 :=
by sorry

end gcd_of_abcd_plus_dcba_l475_47525


namespace bus_ride_difference_l475_47512

theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ) 
  (h1 : oscar_ride = 0.75) (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end bus_ride_difference_l475_47512


namespace time_for_c_is_48_l475_47528

/-- The time it takes for worker c to complete the work alone -/
def time_for_c (time_ab time_bc time_ca : ℚ) : ℚ :=
  let a := (1 / time_ab + 1 / time_ca - 1 / time_bc) / 2
  let b := (1 / time_ab + 1 / time_bc - 1 / time_ca) / 2
  let c := (1 / time_bc + 1 / time_ca - 1 / time_ab) / 2
  1 / c

/-- Theorem stating that given the conditions, c will take 48 days to do the work alone -/
theorem time_for_c_is_48 :
  time_for_c 6 8 12 = 48 := by sorry

end time_for_c_is_48_l475_47528


namespace limit_of_sequence_l475_47585

def a (n : ℕ) : ℚ := (2 - 3 * n^2) / (4 + 5 * n^2)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3/5)| < ε := by sorry

end limit_of_sequence_l475_47585


namespace chemical_mixture_composition_l475_47582

/-- Given the composition of two chemical solutions and their mixture, 
    prove the percentage of chemical b in solution y -/
theorem chemical_mixture_composition 
  (x_a : Real) (x_b : Real) (y_a : Real) (y_b : Real) 
  (mix_a : Real) (mix_x : Real) : 
  x_a = 0.1 → 
  x_b = 0.9 → 
  y_a = 0.2 → 
  mix_a = 0.12 → 
  mix_x = 0.8 → 
  y_b = 0.8 := by
  sorry

#check chemical_mixture_composition

end chemical_mixture_composition_l475_47582


namespace integer_divisibility_l475_47579

theorem integer_divisibility (n : ℕ) (h : ∃ m : ℤ, (2^n - 2 : ℤ) = n * m) :
  ∃ k : ℤ, (2^(2^n - 1) - 2 : ℤ) = (2^n - 1) * k :=
sorry

end integer_divisibility_l475_47579


namespace not_perfect_square_l475_47504

theorem not_perfect_square : 
  ¬ ∃ n : ℕ, 5^2023 = n^2 ∧ 
  ∃ a : ℕ, 3^2021 = a^2 ∧
  ∃ b : ℕ, 7^2024 = b^2 ∧
  ∃ c : ℕ, 6^2025 = c^2 ∧
  ∃ d : ℕ, 8^2026 = d^2 :=
by sorry

end not_perfect_square_l475_47504


namespace greatest_four_digit_divisible_by_63_and_11_l475_47565

def reverse_digits (n : ℕ) : ℕ :=
  sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ m : ℕ,
    is_four_digit m ∧
    is_four_digit (reverse_digits m) ∧
    63 ∣ m ∧
    63 ∣ (reverse_digits m) ∧
    11 ∣ m ∧
    ∀ k : ℕ, (is_four_digit k ∧
              is_four_digit (reverse_digits k) ∧
              63 ∣ k ∧
              63 ∣ (reverse_digits k) ∧
              11 ∣ k) →
              k ≤ m ∧
    m = 9696 :=
  sorry

end greatest_four_digit_divisible_by_63_and_11_l475_47565


namespace machine_operation_l475_47564

theorem machine_operation (x : ℤ) : 26 + x - 6 = 35 → x = 15 := by
  sorry

end machine_operation_l475_47564


namespace small_semicircle_radius_l475_47529

/-- Given a large semicircle with radius 12, a circle with radius 6 inside it, 
    and a smaller semicircle, all pairwise tangent to each other, 
    the radius of the smaller semicircle is 4. -/
theorem small_semicircle_radius (r : ℝ) 
  (h1 : r > 0) -- radius of smaller semicircle is positive
  (h2 : 12 > 0) -- radius of larger semicircle is positive
  (h3 : 6 > 0)  -- radius of circle is positive
  (h4 : r < 12) -- radius of smaller semicircle is less than larger semicircle
  (h5 : r + 6 < 12) -- sum of radii of smaller semicircle and circle is less than larger semicircle
  : r = 4 := by
  sorry

end small_semicircle_radius_l475_47529


namespace selection_schemes_six_four_two_l475_47505

/-- The number of ways to select 4 students from 6 to visit 4 cities,
    where 2 specific students cannot visit one particular city. -/
def selection_schemes (n : ℕ) (k : ℕ) (restricted : ℕ) : ℕ :=
  (n - restricted) * (n - restricted - 1) * (n - 2) * (n - 3)

theorem selection_schemes_six_four_two :
  selection_schemes 6 4 2 = 240 := by
  sorry

end selection_schemes_six_four_two_l475_47505


namespace circle_ratio_l475_47550

/-- Two circles touching externally -/
structure ExternallyTouchingCircles where
  R₁ : ℝ  -- Radius of the first circle
  R₂ : ℝ  -- Radius of the second circle
  h₁ : R₁ > 0
  h₂ : R₂ > 0

/-- Point of tangency between the circles -/
def pointOfTangency (c : ExternallyTouchingCircles) : ℝ := c.R₁ + c.R₂

/-- Distance from point of tangency to center of second circle -/
def tangentDistance (c : ExternallyTouchingCircles) : ℝ := 3 * c.R₂

theorem circle_ratio (c : ExternallyTouchingCircles) 
  (h : tangentDistance c = pointOfTangency c - c.R₁) : 
  c.R₁ = 4 * c.R₂ := by
  sorry

#check circle_ratio

end circle_ratio_l475_47550


namespace tens_digit_of_23_pow_1987_l475_47555

theorem tens_digit_of_23_pow_1987 :
  (23^1987 / 10) % 10 = 4 := by
  sorry

end tens_digit_of_23_pow_1987_l475_47555


namespace petya_wins_against_sasha_l475_47578

/-- Represents a player in the elimination tennis game -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of matches played by each player -/
def matches_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of matches played -/
def total_matches : ℕ := 15

/-- The number of wins by one player against another -/
def wins_against (winner loser : Player) : ℕ := sorry

theorem petya_wins_against_sasha :
  wins_against Player.Petya Player.Sasha = 4 :=
by sorry

end petya_wins_against_sasha_l475_47578


namespace invisible_dots_count_l475_47594

def dice_faces : List Nat := [1, 2, 3, 4, 5, 6]

def visible_faces : List Nat := [6, 5, 3, 1, 4, 2, 1]

def total_faces : Nat := 3 * 6

def visible_faces_count : Nat := 7

def hidden_faces_count : Nat := total_faces - visible_faces_count

theorem invisible_dots_count :
  (3 * (dice_faces.sum)) - (visible_faces.sum) = 41 := by
  sorry

end invisible_dots_count_l475_47594


namespace triangle_radii_inequality_l475_47523

/-- Given a triangle ABC with circumradius R, inradius r, distance from circumcenter to centroid e,
    and distance from incenter to centroid f, prove that R² - e² ≥ 4(r² - f²),
    with equality if and only if the triangle is equilateral. -/
theorem triangle_radii_inequality (R r e f : ℝ) (hR : R > 0) (hr : r > 0) (he : e ≥ 0) (hf : f ≥ 0) :
  R^2 - e^2 ≥ 4*(r^2 - f^2) ∧
  (R^2 - e^2 = 4*(r^2 - f^2) ↔ ∃ (s : ℝ), R = s ∧ r = s/3 ∧ e = s/3 ∧ f = s/6) :=
by sorry

end triangle_radii_inequality_l475_47523


namespace equals_2022_l475_47552

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

end equals_2022_l475_47552


namespace halloween_goodie_bags_minimum_cost_l475_47521

/-- Represents the theme of a Halloween goodie bag -/
inductive Theme
| Vampire
| Pumpkin

/-- Represents the purchase options available -/
inductive PurchaseOption
| Package
| Individual

theorem halloween_goodie_bags_minimum_cost 
  (total_students : ℕ)
  (vampire_requests : ℕ)
  (pumpkin_requests : ℕ)
  (package_price : ℕ)
  (package_size : ℕ)
  (individual_price : ℕ)
  (discount_buy : ℕ)
  (discount_free : ℕ)
  (h1 : total_students = 25)
  (h2 : vampire_requests = 11)
  (h3 : pumpkin_requests = 14)
  (h4 : vampire_requests + pumpkin_requests = total_students)
  (h5 : package_price = 3)
  (h6 : package_size = 5)
  (h7 : individual_price = 1)
  (h8 : discount_buy = 3)
  (h9 : discount_free = 1) :
  (∃ (vampire_packages vampire_individuals pumpkin_packages : ℕ),
    vampire_packages * package_size + vampire_individuals ≥ vampire_requests ∧
    pumpkin_packages * package_size ≥ pumpkin_requests ∧
    (vampire_packages * package_price + vampire_individuals * individual_price +
     (pumpkin_packages / discount_buy * (discount_buy - discount_free) + pumpkin_packages % discount_buy) * package_price = 13)) :=
by sorry


end halloween_goodie_bags_minimum_cost_l475_47521


namespace number_division_problem_l475_47568

theorem number_division_problem (x : ℝ) : x / 5 = 30 + x / 6 ↔ x = 900 := by
  sorry

end number_division_problem_l475_47568


namespace factorization_of_x4_plus_256_l475_47535

theorem factorization_of_x4_plus_256 (x : ℝ) : 
  x^4 + 256 = (x^2 - 8*x + 32) * (x^2 + 8*x + 32) := by
  sorry

end factorization_of_x4_plus_256_l475_47535


namespace multiple_of_17_l475_47542

theorem multiple_of_17 (x y : ℤ) : (2 * x + 3 * y) % 17 = 0 → (9 * x + 5 * y) % 17 = 0 := by
  sorry

end multiple_of_17_l475_47542


namespace cubic_expression_evaluation_l475_47508

theorem cubic_expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 7 = 219 / 7 := by sorry

end cubic_expression_evaluation_l475_47508


namespace unique_prime_p_l475_47533

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 := by
  sorry

end unique_prime_p_l475_47533


namespace tan_420_degrees_l475_47543

theorem tan_420_degrees : Real.tan (420 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_420_degrees_l475_47543


namespace total_leaves_on_our_farm_l475_47563

/-- Represents a farm with trees -/
structure Farm :=
  (num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)

/-- Calculates the total number of leaves on all trees in the farm -/
def total_leaves (f : Farm) : ℕ :=
  f.num_trees * f.branches_per_tree * f.sub_branches_per_branch * f.leaves_per_sub_branch

/-- The farm described in the problem -/
def our_farm : Farm :=
  { num_trees := 4
  , branches_per_tree := 10
  , sub_branches_per_branch := 40
  , leaves_per_sub_branch := 60 }

/-- Theorem stating that the total number of leaves on all trees in our farm is 96,000 -/
theorem total_leaves_on_our_farm : total_leaves our_farm = 96000 := by
  sorry

end total_leaves_on_our_farm_l475_47563


namespace inequality_not_true_l475_47566

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end inequality_not_true_l475_47566


namespace clinton_meal_days_l475_47593

/-- The number of days Clinton buys a meal, given the base cost, up-size cost, and total spent. -/
def days_buying_meal (base_cost : ℚ) (upsize_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (base_cost + upsize_cost)

/-- Theorem stating that Clinton buys the meal for 5 days. -/
theorem clinton_meal_days :
  let base_cost : ℚ := 6
  let upsize_cost : ℚ := 1
  let total_spent : ℚ := 35
  days_buying_meal base_cost upsize_cost total_spent = 5 := by
  sorry

end clinton_meal_days_l475_47593


namespace race_distance_proof_l475_47599

/-- Represents the total distance of a race in meters. -/
def race_distance : ℝ := 88

/-- Represents the time taken by Runner A to complete the race in seconds. -/
def time_A : ℝ := 20

/-- Represents the time taken by Runner B to complete the race in seconds. -/
def time_B : ℝ := 25

/-- Represents the distance by which Runner A beats Runner B in meters. -/
def beating_distance : ℝ := 22

theorem race_distance_proof : 
  race_distance = 88 ∧ 
  (race_distance / time_A) * time_B = race_distance + beating_distance :=
sorry

end race_distance_proof_l475_47599


namespace reflection_of_A_wrt_BC_l475_47540

/-- Reflection of a point with respect to a horizontal line -/
def reflect_point (p : ℝ × ℝ) (y : ℝ) : ℝ × ℝ :=
  (p.1, 2 * y - p.2)

theorem reflection_of_A_wrt_BC :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (0, 1)
  let C : ℝ × ℝ := (3, 1)
  reflect_point A B.2 = (2, -1) := by
sorry

end reflection_of_A_wrt_BC_l475_47540


namespace line_through_point_l475_47531

/-- If the line ax + 3y - 5 = 0 passes through the point (2, 1), then a = 1 -/
theorem line_through_point (a : ℝ) : 
  (a * 2 + 3 * 1 - 5 = 0) → a = 1 := by
  sorry

end line_through_point_l475_47531


namespace jimmy_snow_shoveling_charge_l475_47591

/-- The amount Jimmy charges per driveway for snow shoveling -/
def jimmy_charge_per_driveway : ℝ := 1.50

theorem jimmy_snow_shoveling_charge :
  let candy_bar_price : ℝ := 0.75
  let candy_bar_count : ℕ := 2
  let lollipop_price : ℝ := 0.25
  let lollipop_count : ℕ := 4
  let driveways_shoveled : ℕ := 10
  let candy_store_spend : ℝ := candy_bar_price * candy_bar_count + lollipop_price * lollipop_count
  let snow_shoveling_earnings : ℝ := candy_store_spend * 6
  jimmy_charge_per_driveway = snow_shoveling_earnings / driveways_shoveled :=
by
  sorry

#check jimmy_snow_shoveling_charge

end jimmy_snow_shoveling_charge_l475_47591


namespace drummer_drum_stick_usage_l475_47554

/-- Calculates the total number of drum stick sets used by a drummer over multiple shows. -/
def total_drum_stick_sets (sets_per_show : ℕ) (tossed_sets : ℕ) (num_shows : ℕ) : ℕ :=
  (sets_per_show + tossed_sets) * num_shows

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets, for 30 shows uses 330 sets in total. -/
theorem drummer_drum_stick_usage :
  total_drum_stick_sets 5 6 30 = 330 := by
  sorry

end drummer_drum_stick_usage_l475_47554


namespace nine_integer_lengths_l475_47507

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to the hypotenuse DF in a right triangle DEF -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

theorem nine_integer_lengths (t : RightTriangle) 
  (h1 : t.de = 24) (h2 : t.ef = 25) : 
  countIntegerLengths t = 9 :=
sorry

end nine_integer_lengths_l475_47507


namespace characterization_of_representable_numbers_l475_47501

/-- Two natural numbers are relatively prime if their greatest common divisor is 1 -/
def RelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A natural number k can be represented as the sum of two relatively prime numbers greater than 1 -/
def RepresentableAsSumOfRelativelyPrime (k : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ RelativelyPrime a b ∧ a + b = k

/-- Theorem stating the characterization of numbers representable as sum of two relatively prime numbers greater than 1 -/
theorem characterization_of_representable_numbers :
  ∀ k : ℕ, RepresentableAsSumOfRelativelyPrime k ↔ k = 5 ∨ k ≥ 7 :=
sorry

end characterization_of_representable_numbers_l475_47501


namespace exactly_six_numbers_l475_47561

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧
  ∃ k : ℕ, n - reverse_digits n = k^3 ∧ k > 0

theorem exactly_six_numbers :
  ∃! (s : Finset ℕ), s.card = 6 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end exactly_six_numbers_l475_47561


namespace sum_of_distances_constant_l475_47595

/-- A regular pyramid -/
structure RegularPyramid where
  base : Set (Fin 3 → ℝ)  -- Base of the pyramid as a set of points in ℝ³
  apex : Fin 3 → ℝ        -- Apex of the pyramid as a point in ℝ³
  is_regular : Bool       -- Property ensuring the pyramid is regular

/-- A point on the base of the pyramid -/
def BasePoint (pyramid : RegularPyramid) := { p : Fin 3 → ℝ // p ∈ pyramid.base }

/-- The perpendicular line from a point on the base to the base plane -/
def Perpendicular (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The intersection points of the perpendicular with the face planes -/
def IntersectionPoints (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The sum of distances from a base point to the intersection points -/
def SumOfDistances (pyramid : RegularPyramid) (p : BasePoint pyramid) : ℝ :=
  sorry

/-- Theorem: The sum of distances is constant for all points on the base -/
theorem sum_of_distances_constant (pyramid : RegularPyramid) :
  ∀ p q : BasePoint pyramid, SumOfDistances pyramid p = SumOfDistances pyramid q :=
sorry

end sum_of_distances_constant_l475_47595


namespace mrs_susnas_grade_distribution_l475_47516

/-- Represents the fraction of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ
  passingGrade : ℚ

/-- The actual grade distribution in Mrs. Susna's class -/
def mrsSusnasClass : GradeDistribution where
  b := 1/2
  c := 1/8
  d := 1/12
  f := 1/24
  passingGrade := 7/8
  a := 0  -- We'll prove this value

theorem mrs_susnas_grade_distribution :
  let g := mrsSusnasClass
  g.a + g.b + g.c + g.d + g.f = 1 ∧
  g.a + g.b + g.c = g.passingGrade ∧
  g.a = 1/8 :=
by sorry

end mrs_susnas_grade_distribution_l475_47516


namespace inequality_proof_l475_47589

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_condition : a + b + c = 1) : 
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end inequality_proof_l475_47589


namespace sum_of_rectangle_areas_is_417_l475_47580

/-- The sum of areas of six rectangles with width 3 and lengths (2², 3², 4², 5², 6², 7²) -/
def sum_of_rectangle_areas : ℕ :=
  let width := 3
  let lengths := [2, 3, 4, 5, 6, 7].map (λ x => x^2)
  (lengths.map (λ l => width * l)).sum

/-- Theorem stating that the sum of the areas is 417 -/
theorem sum_of_rectangle_areas_is_417 : sum_of_rectangle_areas = 417 := by
  sorry

end sum_of_rectangle_areas_is_417_l475_47580


namespace geometric_sequence_a7_l475_47572

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 := by
  sorry

end geometric_sequence_a7_l475_47572


namespace runner_speed_ratio_l475_47537

theorem runner_speed_ratio :
  ∀ (v1 v2 : ℝ),
    v1 > v2 →
    v1 - v2 = 4 →
    v1 + v2 = 20 →
    v1 / v2 = 3 / 2 := by
  sorry

end runner_speed_ratio_l475_47537


namespace customer_difference_l475_47538

theorem customer_difference (X Y Z : ℕ) 
  (h1 : X - Y = 10) 
  (h2 : 10 - Z = 4) : 
  X - 4 = 10 := by
  sorry

end customer_difference_l475_47538


namespace quadratic_shift_properties_l475_47587

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Shifts a quadratic function up by a given amount -/
def shift_up (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { f with c := f.c + shift }

theorem quadratic_shift_properties (f : QuadraticFunction) :
  let f_shifted := shift_up f 3
  (f.a = f_shifted.a) ∧ 
  (-f.b / (2 * f.a) = -f_shifted.b / (2 * f_shifted.a)) ∧
  (f.c ≠ f_shifted.c) := by sorry

end quadratic_shift_properties_l475_47587


namespace max_triangles_correct_l475_47511

/-- The maximum number of triangles formed by drawing non-intersecting diagonals in a convex n-gon -/
def max_triangles (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n - 4 else 2 * n - 5

theorem max_triangles_correct (n : ℕ) (h : n ≥ 3) :
  max_triangles n = 
    (if n % 2 = 0 then 2 * n - 4 else 2 * n - 5) ∧
  ∀ k : ℕ, k ≤ max_triangles n :=
by sorry

end max_triangles_correct_l475_47511


namespace fraction_inequality_solution_set_l475_47551

theorem fraction_inequality_solution_set (x : ℝ) :
  (x ≠ -1) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
by sorry

end fraction_inequality_solution_set_l475_47551


namespace rationalize_denominator_l475_47560

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end rationalize_denominator_l475_47560


namespace base4_10201_to_decimal_l475_47586

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10201_to_decimal :
  base4_to_decimal [1, 0, 2, 0, 1] = 289 := by
  sorry

end base4_10201_to_decimal_l475_47586


namespace friends_bill_calculation_l475_47556

/-- Represents a restaurant order --/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order --/
def cost (o : Order) (taco_price enchilada_price : ℚ) : ℚ :=
  o.tacos * taco_price + o.enchiladas * enchilada_price

theorem friends_bill_calculation (enchilada_price : ℚ) 
  (your_order friend_order : Order) (your_bill : ℚ) 
  (h1 : enchilada_price = 2)
  (h2 : your_order = ⟨2, 3⟩)
  (h3 : friend_order = ⟨3, 5⟩)
  (h4 : your_bill = 39/5) : 
  ∃ (taco_price : ℚ), cost friend_order taco_price enchilada_price = 127/10 := by
  sorry

#eval 127/10  -- Should output 12.7

end friends_bill_calculation_l475_47556


namespace arithmetic_sequence_sum_specific_l475_47575

def arithmetic_sequence_sum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-45) 3 4 = -273 :=
by sorry

end arithmetic_sequence_sum_specific_l475_47575


namespace square_circle_area_ratio_l475_47517

/-- Given a square and a circle intersecting such that each side of the square contains
    a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 3/π. -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * Real.sqrt 3
  (s^2) / (π * r^2) = 3 / π :=
by sorry

end square_circle_area_ratio_l475_47517


namespace largest_common_value_less_than_1000_l475_47592

theorem largest_common_value_less_than_1000 :
  let seq1 := {a : ℕ | ∃ n : ℕ, a = 2 + 3 * n}
  let seq2 := {a : ℕ | ∃ m : ℕ, a = 4 + 8 * m}
  let common_values := seq1 ∩ seq2
  (∃ x ∈ common_values, x < 1000 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) →
  (∃ x ∈ common_values, x = 980 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) :=
by sorry

end largest_common_value_less_than_1000_l475_47592


namespace alpha_necessary_not_sufficient_for_beta_l475_47596

-- Define the propositions
def α (x : ℝ) : Prop := |x - 1| ≤ 2
def β (x : ℝ) : Prop := (x - 3) / (x + 1) ≤ 0

-- Theorem statement
theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x, β x → α x) ∧ ¬(∀ x, α x → β x) := by sorry

end alpha_necessary_not_sufficient_for_beta_l475_47596


namespace fourth_term_of_sequence_l475_47598

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) 
    (h1 : a 1 = 2)
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n) :
  a 4 = 54 := by
  sorry

end fourth_term_of_sequence_l475_47598


namespace adidas_cost_l475_47510

/-- The cost of Adidas shoes given the sales information -/
theorem adidas_cost (total_goal : ℝ) (nike_cost reebok_cost : ℝ) 
  (nike_sold adidas_sold reebok_sold : ℕ) (excess : ℝ) 
  (h1 : total_goal = 1000)
  (h2 : nike_cost = 60)
  (h3 : reebok_cost = 35)
  (h4 : nike_sold = 8)
  (h5 : adidas_sold = 6)
  (h6 : reebok_sold = 9)
  (h7 : excess = 65)
  : ∃ (adidas_cost : ℝ), 
    nike_cost * nike_sold + adidas_cost * adidas_sold + reebok_cost * reebok_sold 
    = total_goal + excess ∧ adidas_cost = 45 := by
  sorry

end adidas_cost_l475_47510


namespace smallest_possible_d_value_l475_47524

theorem smallest_possible_d_value (d : ℝ) : 
  (2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2 = (4 * d) ^ 2 → 
  d ≥ (1 + 2 * Real.sqrt 10) / 3 :=
by sorry

end smallest_possible_d_value_l475_47524


namespace unique_starting_digit_l475_47549

def starts_with (x : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, d * 10^k ≤ x ∧ x < (d + 1) * 10^k

theorem unique_starting_digit :
  ∃! a : ℕ, a < 10 ∧ 
    (∃ n : ℕ, starts_with (2^n) a ∧ starts_with (5^n) a) ∧
    (a^2 < 10 ∧ 10 < (a+1)^2) :=
by sorry

end unique_starting_digit_l475_47549


namespace wall_width_is_three_l475_47513

/-- Proves that a rectangular wall with given proportions and volume has a width of 3 meters -/
theorem wall_width_is_three (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
  sorry

end wall_width_is_three_l475_47513


namespace parallelepiped_arrangement_exists_l475_47558

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  -- Define the parallelepiped structure (simplified for this example)
  dummy : Unit

/-- Represents a point in 3D space -/
structure Point where
  -- Define the point structure (simplified for this example)
  dummy : Unit

/-- Checks if two parallelepipeds intersect -/
def intersects (p1 p2 : Parallelepiped) : Prop :=
  sorry

/-- Checks if a point is inside a parallelepiped -/
def isInside (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Checks if a vertex of a parallelepiped is visible from a point -/
def isVertexVisible (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Theorem stating the existence of the required arrangement -/
theorem parallelepiped_arrangement_exists : 
  ∃ (parallelepipeds : Fin 6 → Parallelepiped) (observationPoint : Point),
    (∀ i j : Fin 6, i ≠ j → ¬intersects (parallelepipeds i) (parallelepipeds j)) ∧
    (∀ i : Fin 6, ¬isInside observationPoint (parallelepipeds i)) ∧
    (∀ i : Fin 6, ¬isVertexVisible observationPoint (parallelepipeds i)) :=
  sorry

end parallelepiped_arrangement_exists_l475_47558


namespace quadratic_rewrite_ratio_l475_47557

theorem quadratic_rewrite_ratio : ∃ (c r s : ℚ),
  (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + r)^2 + s) ∧
  s / r = -62 / 3 := by
  sorry

end quadratic_rewrite_ratio_l475_47557


namespace remainder_theorem_l475_47509

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_theorem_l475_47509


namespace reflection_changes_color_l475_47527

/-- Determines if a number is red (can be expressed as 81x + 100y for positive integers x and y) -/
def isRed (n : ℤ) : Prop :=
  ∃ x y : ℕ+, n = 81 * x + 100 * y

/-- The point P -/
def P : ℤ := 3960

/-- Reflects a point T with respect to P -/
def reflect (T : ℤ) : ℤ := 2 * P - T

theorem reflection_changes_color :
  ∀ T : ℤ, isRed T ≠ isRed (reflect T) :=
sorry

end reflection_changes_color_l475_47527


namespace quarter_percentage_approx_l475_47539

/-- Represents the number and value of coins -/
structure Coins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ
  dime_value : ℕ
  quarter_value : ℕ
  nickel_value : ℕ

/-- Calculates the percentage of quarters in the total value -/
def quarter_percentage (c : Coins) : ℚ :=
  let total_value := c.dimes * c.dime_value + c.quarters * c.quarter_value + c.nickels * c.nickel_value
  let quarter_value := c.quarters * c.quarter_value
  (quarter_value : ℚ) / (total_value : ℚ) * 100

/-- Theorem stating that the percentage of quarters is approximately 51.28% -/
theorem quarter_percentage_approx (c : Coins) 
  (h1 : c.dimes = 80) (h2 : c.quarters = 40) (h3 : c.nickels = 30)
  (h4 : c.dime_value = 10) (h5 : c.quarter_value = 25) (h6 : c.nickel_value = 5) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ |quarter_percentage c - (5128 : ℚ) / 100| < ε := by
  sorry

end quarter_percentage_approx_l475_47539


namespace juan_tricycles_l475_47577

theorem juan_tricycles (cars bicycles pickups : ℕ) (total_tires : ℕ) : 
  cars = 15 → 
  bicycles = 3 → 
  pickups = 8 → 
  total_tires = 101 → 
  ∃ (tricycles : ℕ), 
    cars * 4 + bicycles * 2 + pickups * 4 + tricycles * 3 = total_tires ∧ 
    tricycles = 1 := by
  sorry

end juan_tricycles_l475_47577


namespace cubic_function_max_l475_47520

/-- Given a cubic function with specific properties, prove its maximum value on [-3, 3] -/
theorem cubic_function_max (a b c : ℝ) : 
  (∀ x, (∃ y, y = a * x^3 + b * x + c)) →  -- f(x) = ax³ + bx + c
  (∃ y, y = 8 * a + 2 * b + c ∧ y = c - 16) →  -- f(2) = c - 16
  (3 * a * 2^2 + b = 0) →  -- f'(2) = 0 (extremum condition)
  (a = 1 ∧ b = -12) →  -- Values of a and b
  (∃ x, ∀ y, a * x^3 + b * x + c ≥ y ∧ a * x^3 + b * x + c = 28) →  -- Maximum value is 28
  (∃ x, x ∈ Set.Icc (-3) 3 ∧ 
    ∀ y ∈ Set.Icc (-3) 3, a * x^3 + b * x + c ≥ a * y^3 + b * y + c ∧ 
    a * x^3 + b * x + c = 28) :=
by sorry

end cubic_function_max_l475_47520


namespace tournament_games_l475_47570

theorem tournament_games (x : ℕ) : 
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * x = x ∧ 
  (2 / 3 : ℚ) * (x + 10) + (1 / 3 : ℚ) * (x + 10) = x + 10 ∧
  (2 / 3 : ℚ) * (x + 10) = (3 / 4 : ℚ) * x + 5 ∧
  (1 / 3 : ℚ) * (x + 10) = (1 / 4 : ℚ) * x + 5 →
  x = 20 := by
sorry

end tournament_games_l475_47570


namespace sin_2theta_plus_pi_third_l475_47515

theorem sin_2theta_plus_pi_third (θ : Real) 
  (h1 : θ > π / 2) (h2 : θ < π) 
  (h3 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.sin (2 * θ + π / 3) = 1 / 2 := by
  sorry

end sin_2theta_plus_pi_third_l475_47515


namespace inequality_holds_l475_47574

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 - x*y - x*z - y*z ≥ 0 := by
  sorry

end inequality_holds_l475_47574


namespace chess_game_draw_probability_l475_47506

theorem chess_game_draw_probability (P_A_not_losing P_B_not_losing : ℝ) 
  (h1 : P_A_not_losing = 0.8)
  (h2 : P_B_not_losing = 0.7)
  (h3 : ∀ P_A_win P_B_win P_draw : ℝ, 
    P_A_win + P_draw = P_A_not_losing → 
    P_B_win + P_draw = P_B_not_losing → 
    P_A_win + P_B_win + P_draw = 1 → 
    P_draw = 0.5) :
  ∃ P_draw : ℝ, P_draw = 0.5 := by
sorry

end chess_game_draw_probability_l475_47506


namespace square_clock_area_l475_47518

-- Define the side length of the square clock
def clock_side_length : ℝ := 30

-- Define the area of the square clock
def clock_area : ℝ := clock_side_length * clock_side_length

-- Theorem to prove
theorem square_clock_area : clock_area = 900 := by
  sorry

end square_clock_area_l475_47518


namespace student_pairs_count_l475_47534

def number_of_students : ℕ := 15

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem student_pairs_count :
  choose number_of_students 2 = 105 := by
  sorry

end student_pairs_count_l475_47534


namespace grass_seed_min_cost_l475_47581

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat

/-- Finds the minimum cost to buy grass seed given the constraints -/
def minCostGrassSeed (bags : List GrassSeedBag) (minWeight maxWeight : Nat) : Rat :=
  sorry

/-- Theorem stating the minimum cost for the given problem -/
theorem grass_seed_min_cost :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 138/10 },
    { weight := 10, price := 2043/100 },
    { weight := 25, price := 3225/100 }
  ]
  minCostGrassSeed bags 65 80 = 9675/100 := by sorry

end grass_seed_min_cost_l475_47581


namespace marble_fraction_after_tripling_l475_47503

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (2/3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3/5 := by
sorry

end marble_fraction_after_tripling_l475_47503


namespace cookies_calculation_l475_47532

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 25

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 45

/-- The total number of cookies Brenda's mother prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation :
  total_cookies = 1125 :=
by sorry

end cookies_calculation_l475_47532


namespace not_all_even_P_true_l475_47546

/-- A proposition P on even natural numbers -/
def P : ℕ → Prop := sorry

/-- Theorem stating that we cannot conclude P holds for all even natural numbers -/
theorem not_all_even_P_true :
  (∀ n : ℕ, n ≤ 1001 → P (2 * n)) →
  ¬(∀ k : ℕ, Even k → P k) :=
by sorry

end not_all_even_P_true_l475_47546


namespace polynomial_multiplication_l475_47545

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^4 - 4 * y^3) * (9 * x^8 + 12 * x^4 * y^3 + 16 * y^6) = 27 * x^12 - 64 * y^9 := by
  sorry

end polynomial_multiplication_l475_47545


namespace product_digits_sum_base7_l475_47553

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

theorem product_digits_sum_base7 :
  let a := 35
  let b := 42
  let product := toBase7 (toDecimal a * toDecimal b)
  sumDigitsBase7 product = 15
  := by sorry

end product_digits_sum_base7_l475_47553


namespace sum_bounds_l475_47522

def A : Set ℕ := {n | n ≤ 2018}

theorem sum_bounds (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) 
  (h : x^2 + y^2 - z^2 = 2019^2) : 
  2181 ≤ x + y + z ∧ x + y + z ≤ 5781 := by
  sorry

end sum_bounds_l475_47522


namespace magic_king_seasons_l475_47571

theorem magic_king_seasons (total_episodes : ℕ) 
  (episodes_first_half : ℕ) (episodes_second_half : ℕ) :
  total_episodes = 225 ∧ 
  episodes_first_half = 20 ∧ 
  episodes_second_half = 25 →
  ∃ (seasons : ℕ), 
    seasons = 10 ∧
    total_episodes = (seasons / 2) * episodes_first_half + 
                     (seasons / 2) * episodes_second_half :=
by sorry

end magic_king_seasons_l475_47571


namespace smallest_good_sequence_index_is_60_l475_47536

-- Define a good sequence
def GoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a 0 = k) ∧
  (∀ i : ℕ, (a (i + 1) = 2 * a i + 1) ∨ (a (i + 1) = a i / (a i + 2))) ∧
  (∃ k : ℕ+, a k = 2014)

-- Define the property we want to prove
def SmallestGoodSequenceIndex : Prop :=
  ∃ n : ℕ+, 
    (∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) ∧
    (∀ m : ℕ+, m < n → ¬∃ a : ℕ → ℝ, GoodSequence a ∧ a m = 2014)

-- The theorem to prove
theorem smallest_good_sequence_index_is_60 : 
  SmallestGoodSequenceIndex ∧ (∃ n : ℕ+, n = 60 ∧ ∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) :=
sorry

end smallest_good_sequence_index_is_60_l475_47536


namespace distance_point_to_line_polar_example_l475_47597

/-- The distance from a point to a line in polar coordinates -/
def distance_point_to_line_polar (ρ₀ : ℝ) (θ₀ : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  sorry

theorem distance_point_to_line_polar_example :
  distance_point_to_line_polar 2 (π/3) (fun ρ θ ↦ ρ * Real.cos (θ + π/3) - 2) = 3 :=
sorry

end distance_point_to_line_polar_example_l475_47597


namespace cookies_per_bag_l475_47573

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 2173) (h2 : num_bags = 53) :
  total_cookies / num_bags = 41 := by
  sorry

end cookies_per_bag_l475_47573


namespace circle_area_above_line_is_zero_l475_47500

/-- The circle equation: x^2 - 8x + y^2 - 10y + 29 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 29 = 0

/-- The line equation: y = x - 2 -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 2

/-- The area of the circle above the line -/
def area_above_line (circle : (ℝ × ℝ) → Prop) (line : (ℝ × ℝ) → Prop) : ℝ :=
  sorry -- Definition of area calculation

theorem circle_area_above_line_is_zero :
  area_above_line (λ (x, y) ↦ circle_equation x y) (λ (x, y) ↦ line_equation x y) = 0 := by
  sorry

end circle_area_above_line_is_zero_l475_47500


namespace regular_polygon_with_740_diagonals_has_40_sides_l475_47530

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 740 diagonals has 40 sides -/
theorem regular_polygon_with_740_diagonals_has_40_sides :
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = 740 ∧ n = 40 := by
  sorry

end regular_polygon_with_740_diagonals_has_40_sides_l475_47530


namespace range_of_f_l475_47526

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l475_47526


namespace layer_sum_2014_implies_digit_sum_13_l475_47576

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  w : Nat
  x : Nat
  y : Nat
  z : Nat
  w_nonzero : w ≠ 0
  w_upper_bound : w < 10
  x_upper_bound : x < 10
  y_upper_bound : y < 10
  z_upper_bound : z < 10

/-- Calculates the layer sum of a four-digit integer --/
def layerSum (n : FourDigitInt) : Nat :=
  1000 * n.w + 100 * n.x + 10 * n.y + n.z +
  100 * n.x + 10 * n.y + n.z +
  10 * n.y + n.z +
  n.z

/-- Main theorem --/
theorem layer_sum_2014_implies_digit_sum_13 (n : FourDigitInt) :
  layerSum n = 2014 → n.w + n.x + n.y + n.z = 13 := by
  sorry

end layer_sum_2014_implies_digit_sum_13_l475_47576


namespace different_terminal_sides_l475_47541

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- Theorem statement
theorem different_terminal_sides :
  ¬ same_terminal_side 1050 (-300) :=
by
  sorry

end different_terminal_sides_l475_47541


namespace age_difference_l475_47590

/-- Represents the ages of a mother and daughter pair -/
structure AgesPair where
  mother : ℕ
  daughter : ℕ

/-- Checks if the digits of the daughter's age are the reverse of the mother's age digits -/
def AgesPair.isReverse (ages : AgesPair) : Prop :=
  ages.daughter = ages.mother % 10 * 10 + ages.mother / 10

/-- Checks if in 13 years, the mother will be twice as old as the daughter -/
def AgesPair.futureCondition (ages : AgesPair) : Prop :=
  ages.mother + 13 = 2 * (ages.daughter + 13)

/-- The main theorem stating the age difference -/
theorem age_difference (ages : AgesPair) 
  (h1 : ages.isReverse)
  (h2 : ages.futureCondition) :
  ages.mother - ages.daughter = 40 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (ages : AgesPair), ages.isReverse ∧ ages.futureCondition ∧ ages.mother - ages.daughter = 40 := by
  sorry

end age_difference_l475_47590


namespace triangle_area_l475_47569

/-- Given a triangle with perimeter 40 and inradius 2.5, prove its area is 50 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 40 → r = 2.5 → A = r * (P / 2) → A = 50 := by sorry

end triangle_area_l475_47569


namespace construction_labor_cost_l475_47514

def worker_salary : ℕ := 100
def electrician_salary : ℕ := 2 * worker_salary
def plumber_salary : ℕ := (5 * worker_salary) / 2
def architect_salary : ℕ := (7 * worker_salary) / 2

def project_cost : ℕ := 2 * worker_salary + electrician_salary + plumber_salary + architect_salary

def total_cost : ℕ := 3 * project_cost

theorem construction_labor_cost : total_cost = 3000 := by
  sorry

end construction_labor_cost_l475_47514
