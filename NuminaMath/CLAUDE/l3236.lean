import Mathlib

namespace special_polynomial_is_x_squared_plus_one_l3236_323669

/-- A polynomial satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p (x * y) = p x * p y - p x - p y + 2) ∧
  p 3 = 10 ∧
  p 4 = 17

/-- The theorem stating that the special polynomial is x^2 + 1 -/
theorem special_polynomial_is_x_squared_plus_one 
  (p : ℝ → ℝ) (hp : SpecialPolynomial p) :
  ∀ x : ℝ, p x = x^2 + 1 := by
  sorry

end special_polynomial_is_x_squared_plus_one_l3236_323669


namespace tangent_and_inequality_l3236_323612

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

theorem tangent_and_inequality (m n : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ f m x = n) →  -- Point N(1, n) on the curve
  (∃ (x : ℝ), x = 1 ∧ (deriv (f m)) x = 1) →  -- Tangent with slope 1 (tan(π/4)) at x = 1
  (m = 2/3 ∧ n = -1/3) ∧  -- Part 1 of the theorem
  (∃ (k : ℕ), k = 2008 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f m x ≤ k - 1993 ∧
    ∀ (k' : ℕ), k' < k → ∃ (x : ℝ), x ∈ Set.Icc (-1) 3 ∧ f m x > k' - 1993) :=
by sorry

end tangent_and_inequality_l3236_323612


namespace quadratic_inequality_solution_set_l3236_323614

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_solution_set_l3236_323614


namespace symmetric_points_product_l3236_323678

/-- Given two points A(-2, a) and B(b, -3) symmetric about the y-axis, prove that ab = -6 -/
theorem symmetric_points_product (a b : ℝ) : 
  ((-2 : ℝ) = -b) → (a = -3) → ab = -6 := by sorry

end symmetric_points_product_l3236_323678


namespace fraction_equality_l3236_323642

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) ↔ x = -1 := by
  sorry

end fraction_equality_l3236_323642


namespace problem_hall_tilings_l3236_323601

/-- Represents a tiling configuration for a rectangular hall. -/
structure HallTiling where
  width : ℕ
  length : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Counts the number of valid tiling configurations. -/
def countValidTilings (h : HallTiling) : ℕ :=
  sorry

/-- The specific hall configuration from the problem. -/
def problemHall : HallTiling :=
  { width := 2
  , length := 13
  , black_tiles := 11
  , white_tiles := 15 }

/-- Theorem stating that the number of valid tilings for the problem hall is 486. -/
theorem problem_hall_tilings :
  countValidTilings problemHall = 486 :=
sorry

end problem_hall_tilings_l3236_323601


namespace tan_alpha_4_implies_fraction_9_l3236_323629

theorem tan_alpha_4_implies_fraction_9 (α : Real) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end tan_alpha_4_implies_fraction_9_l3236_323629


namespace left_translation_exponential_l3236_323628

/-- Given a function f: ℝ → ℝ, we say it's a left translation by 2 units of g 
    if f(x) = g(x + 2) for all x ∈ ℝ -/
def is_left_translation_by_two (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2)

/-- The theorem stating that if f is a left translation by 2 units of the function
    x ↦ 2^(2x-1), then f(x) = 2^(2x-5) for all x ∈ ℝ -/
theorem left_translation_exponential 
  (f : ℝ → ℝ) 
  (h : is_left_translation_by_two f (fun x ↦ 2^(2*x - 1))) :
  ∀ x, f x = 2^(2*x - 5) := by
  sorry

end left_translation_exponential_l3236_323628


namespace statistics_collection_count_l3236_323672

/-- Represents the multiset of letters in "STATISTICS" --/
def statistics : Multiset Char := {'S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S'}

/-- Represents the vowels in "STATISTICS" --/
def vowels : Multiset Char := {'A', 'I', 'I'}

/-- Represents the consonants in "STATISTICS", with S and T treated as indistinguishable --/
def consonants : Multiset Char := {'C', 'S', 'S', 'S'}

/-- The number of distinct collections of 7 letters (3 vowels and 4 consonants) from "STATISTICS" --/
def distinct_collections : ℕ := 30

theorem statistics_collection_count :
  (Multiset.card statistics = 10) →
  (Multiset.card vowels = 3) →
  (Multiset.card consonants = 4) →
  (∀ x ∈ vowels, x ∈ statistics) →
  (∀ x ∈ consonants, x ∈ statistics ∨ x = 'S') →
  (distinct_collections = 30) := by
  sorry

end statistics_collection_count_l3236_323672


namespace complex_on_line_l3236_323690

theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → a = 0 := by
sorry

end complex_on_line_l3236_323690


namespace polynomial_divisibility_l3236_323626

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := by
sorry

end polynomial_divisibility_l3236_323626


namespace complex_magnitude_equation_l3236_323683

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 2 * Complex.I * Real.sqrt 3) * Complex.abs (6 - 4 * Complex.I) = 26) → t = 1 := by
  sorry

end complex_magnitude_equation_l3236_323683


namespace mixture_replacement_theorem_l3236_323634

/-- The amount of mixture replaced to change the ratio from 7:5 to 7:9 -/
def mixture_replaced (initial_total : ℝ) (replaced : ℝ) : Prop :=
  let initial_a := 21
  let initial_b := initial_total - initial_a
  let new_b := initial_b + replaced
  (initial_a / initial_total = 7 / 12) ∧
  (initial_a / new_b = 7 / 9) ∧
  replaced = 12

/-- Theorem stating that 12 liters of mixture were replaced -/
theorem mixture_replacement_theorem :
  ∃ (initial_total : ℝ), mixture_replaced initial_total 12 := by
  sorry

end mixture_replacement_theorem_l3236_323634


namespace square_perimeter_side_length_l3236_323680

theorem square_perimeter_side_length (perimeter : ℝ) (side : ℝ) : 
  perimeter = 8 → side ≠ 4 := by
  sorry

end square_perimeter_side_length_l3236_323680


namespace largest_digit_divisible_by_six_l3236_323656

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end largest_digit_divisible_by_six_l3236_323656


namespace trick_sum_prediction_l3236_323605

theorem trick_sum_prediction (a b : ℕ) (ha : 10000 ≤ a ∧ a < 100000) : 
  a + b + (99999 - b) = 100000 + a - 1 := by
  sorry

end trick_sum_prediction_l3236_323605


namespace smallest_positive_integer_with_remainders_l3236_323608

theorem smallest_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 3 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 6 = 5 → n ≤ m) ∧
  n = 11 := by
sorry

end smallest_positive_integer_with_remainders_l3236_323608


namespace initial_bees_in_hive_l3236_323641

theorem initial_bees_in_hive (additional_bees : ℕ) (total_bees : ℕ) (h1 : additional_bees = 9) (h2 : total_bees = 25) :
  total_bees - additional_bees = 16 := by
  sorry

end initial_bees_in_hive_l3236_323641


namespace line_equations_proof_l3236_323617

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if two lines are perpendicular -/
def Line.isPerpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations_proof :
  let l1 : Line := { a := 3, b := -2, c := 1 }
  let l2 : Line := { a := 3, b := -2, c := 5 }
  let l3 : Line := { a := 3, b := -2, c := -5 }
  let l4 : Line := { a := 2, b := 3, c := 1 }
  (l1.containsPoint 1 2 ∧ l1.isParallelTo l2) ∧
  (l3.containsPoint 1 (-1) ∧ l3.isPerpendicularTo l4) := by sorry

end line_equations_proof_l3236_323617


namespace inequality_holds_iff_even_l3236_323653

theorem inequality_holds_iff_even (n : ℕ+) :
  (∀ x : ℝ, 3 * x^(n : ℕ) + n * (x + 2) - 3 ≥ n * x^2) ↔ Even n := by
  sorry

end inequality_holds_iff_even_l3236_323653


namespace white_balls_count_l3236_323660

theorem white_balls_count 
  (total_balls : ℕ) 
  (total_draws : ℕ) 
  (white_draws : ℕ) 
  (h1 : total_balls = 20) 
  (h2 : total_draws = 404) 
  (h3 : white_draws = 101) : 
  (total_balls : ℚ) * (white_draws : ℚ) / (total_draws : ℚ) = 5 := by
  sorry

end white_balls_count_l3236_323660


namespace min_value_reciprocal_sum_l3236_323603

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 :=
by sorry

end min_value_reciprocal_sum_l3236_323603


namespace sequence_equality_l3236_323682

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- The main theorem to prove -/
theorem sequence_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end sequence_equality_l3236_323682


namespace quadratic_solution_l3236_323657

theorem quadratic_solution (y : ℝ) : 
  y > 0 ∧ 6 * y^2 + 5 * y - 12 = 0 ↔ y = (-5 + Real.sqrt 313) / 12 := by
  sorry

end quadratic_solution_l3236_323657


namespace cylinder_volume_l3236_323686

/-- Given a sphere and a cylinder with specific properties, prove the volume of the cylinder --/
theorem cylinder_volume (sphere_volume : ℝ) (cylinder_base_diameter : ℝ) :
  sphere_volume = (500 * Real.pi) / 3 →
  cylinder_base_diameter = 8 →
  ∃ (cylinder_volume : ℝ),
    cylinder_volume = 96 * Real.pi ∧
    (∃ (sphere_radius : ℝ) (cylinder_height : ℝ),
      (4 / 3) * Real.pi * sphere_radius ^ 3 = sphere_volume ∧
      cylinder_height ^ 2 = sphere_radius ^ 2 - (cylinder_base_diameter / 2) ^ 2 ∧
      cylinder_volume = Real.pi * (cylinder_base_diameter / 2) ^ 2 * cylinder_height) :=
by sorry


end cylinder_volume_l3236_323686


namespace current_speed_current_speed_is_3_l3236_323644

/-- The speed of the current in a river, given the man's rowing speed in still water,
    the distance covered downstream, and the time taken to cover that distance. -/
theorem current_speed (mans_speed : ℝ) (distance : ℝ) (time : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - mans_speed

/-- Proof that the speed of the current is 3 kmph -/
theorem current_speed_is_3 :
  current_speed 15 0.06 11.999040076793857 = 3 := by
  sorry

end current_speed_current_speed_is_3_l3236_323644


namespace number_of_pupils_in_class_l3236_323699

/-- 
Given a class where:
1. A pupil's marks were wrongly entered as 67 instead of 45.
2. The wrong entry caused the average marks for the class to increase by half a mark.

Prove that the number of pupils in the class is 44.
-/
theorem number_of_pupils_in_class : ℕ := by
  sorry

end number_of_pupils_in_class_l3236_323699


namespace total_spent_l3236_323615

def trick_deck_price : ℕ := 8
def victor_decks : ℕ := 6
def friend_decks : ℕ := 2

theorem total_spent : 
  trick_deck_price * victor_decks + trick_deck_price * friend_decks = 64 := by
  sorry

end total_spent_l3236_323615


namespace monic_quartic_polynomial_value_at_zero_l3236_323643

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value_at_zero 
  (f : ℝ → ℝ) 
  (hf : MonicQuarticPolynomial f) 
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f 3 = -9)
  (h4 : f 5 = -25) :
  f 0 = -30 := by
sorry

end monic_quartic_polynomial_value_at_zero_l3236_323643


namespace correct_addition_and_rounding_l3236_323604

-- Define the addition operation
def add (a b : ℕ) : ℕ := a + b

-- Define the rounding operation to the nearest ten
def roundToNearestTen (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  if lastDigit < 5 then
    n - lastDigit
  else
    n + (10 - lastDigit)

-- Theorem statement
theorem correct_addition_and_rounding :
  roundToNearestTen (add 46 37) = 80 := by sorry

end correct_addition_and_rounding_l3236_323604


namespace farmer_cows_problem_l3236_323633

theorem farmer_cows_problem (initial_food : ℝ) (initial_cows : ℕ) :
  initial_food > 0 →
  initial_cows > 0 →
  (initial_food / 50 = initial_food / (5 * 10)) →
  (initial_cows = 200) :=
by
  sorry

end farmer_cows_problem_l3236_323633


namespace other_communities_count_l3236_323649

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 238 := by
  sorry

end other_communities_count_l3236_323649


namespace largest_c_value_l3236_323658

theorem largest_c_value (c : ℝ) (h : (3*c + 4)*(c - 2) = 9*c) : 
  c ≤ 4 ∧ ∃ (c : ℝ), (3*c + 4)*(c - 2) = 9*c ∧ c = 4 := by
  sorry

end largest_c_value_l3236_323658


namespace inequality_proof_l3236_323674

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end inequality_proof_l3236_323674


namespace equation_solution_l3236_323625

theorem equation_solution (x y : ℝ) : 
  x / 3 - y / 2 = 1 → y = 2 * x / 3 - 2 := by
  sorry

end equation_solution_l3236_323625


namespace divisibility_sequence_l3236_323697

theorem divisibility_sequence (t : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ (p : ℤ) ∣ ((3 - 7*t) * 2^n + (18*t - 9) * 3^n + (6 - 10*t) * 4^n) := by
  sorry

end divisibility_sequence_l3236_323697


namespace tangent_equation_solution_l3236_323646

open Real

theorem tangent_equation_solution (x : ℝ) :
  (5.44 * tan (5 * x) - 2 * tan (3 * x) = tan (3 * x)^2 * tan (5 * x)) →
  (cos (3 * x) ≠ 0) →
  (cos (5 * x) ≠ 0) →
  ∃ k : ℤ, x = π * k := by
sorry

end tangent_equation_solution_l3236_323646


namespace chopped_cube_height_l3236_323652

/-- Given a unit cube with a corner chopped off through the midpoints of the three adjacent edges,
    when the freshly-cut face is placed on a table, the height of the remaining solid is 29/32. -/
theorem chopped_cube_height : 
  let cube_edge : ℝ := 1
  let midpoint_factor : ℝ := 1/2
  let chopped_volume : ℝ := 3/32
  let remaining_volume : ℝ := 1 - chopped_volume
  let base_area : ℝ := cube_edge^2
  remaining_volume / base_area = 29/32 := by sorry

end chopped_cube_height_l3236_323652


namespace flea_collar_count_l3236_323655

/-- Represents the number of dogs with flea collars in a kennel -/
def dogs_with_flea_collars (total : ℕ) (with_tags : ℕ) (with_both : ℕ) (with_neither : ℕ) : ℕ :=
  total - with_tags + with_both - with_neither

/-- Theorem stating that in a kennel of 80 dogs, where 45 dogs wear tags, 
    6 dogs wear both tags and flea collars, and 1 dog wears neither, 
    the number of dogs wearing flea collars is 40. -/
theorem flea_collar_count : 
  dogs_with_flea_collars 80 45 6 1 = 40 := by
  sorry

end flea_collar_count_l3236_323655


namespace unique_solution_implies_n_equals_8_l3236_323663

-- Define the quadratic equation
def quadratic_equation (n : ℝ) (x : ℝ) : ℝ := 4 * x^2 + n * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (n : ℝ) : ℝ := n^2 - 4 * 4 * 4

-- Theorem statement
theorem unique_solution_implies_n_equals_8 :
  ∃! x : ℝ, quadratic_equation 8 x = 0 ∧
  ∀ n : ℝ, (∃! x : ℝ, quadratic_equation n x = 0) → n = 8 ∨ n = -8 :=
by sorry

end unique_solution_implies_n_equals_8_l3236_323663


namespace basketball_game_probability_basketball_game_probability_proof_l3236_323665

/-- The probability that at least 7 out of 8 people stay for an entire basketball game,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem basketball_game_probability : ℝ :=
  let total_people : ℕ := 8
  let certain_people : ℕ := 4
  let uncertain_people : ℕ := 4
  let stay_probability : ℝ := 1/3
  let at_least_stay : ℕ := 7

  1/9

/-- Proof of the basketball game probability theorem -/
theorem basketball_game_probability_proof :
  basketball_game_probability = 1/9 := by
  sorry

end basketball_game_probability_basketball_game_probability_proof_l3236_323665


namespace original_eq_hyperbola_and_ellipse_l3236_323651

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 4*x^4 = 2*y^2 - 1

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 2*x^2 = 1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 2*x^2 = 1

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
by sorry

end original_eq_hyperbola_and_ellipse_l3236_323651


namespace bolt_nut_balance_l3236_323696

theorem bolt_nut_balance (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) (bolt_workers : ℕ) : 
  total_workers = 56 →
  bolts_per_worker = 16 →
  nuts_per_worker = 24 →
  nuts_per_bolt = 2 →
  bolt_workers ≤ total_workers →
  (2 * bolts_per_worker * bolt_workers = nuts_per_worker * (total_workers - bolt_workers)) ↔
  (bolts_per_worker * bolt_workers * nuts_per_bolt = nuts_per_worker * (total_workers - bolt_workers)) :=
by sorry

end bolt_nut_balance_l3236_323696


namespace alpha_value_l3236_323648

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = -16 - 3 * Complex.I := by
  sorry

end alpha_value_l3236_323648


namespace dividend_calculation_l3236_323688

theorem dividend_calculation (D d Q R : ℕ) 
  (eq_condition : D = d * Q + R)
  (d_value : d = 17)
  (Q_value : Q = 9)
  (R_value : R = 9) :
  D = 162 := by
sorry

end dividend_calculation_l3236_323688


namespace homework_problem_l3236_323681

theorem homework_problem (p t : ℕ) : 
  p ≥ 10 ∧ 
  p * t = (2 * p + 2) * (t + 1) →
  p * t = 60 :=
by sorry

end homework_problem_l3236_323681


namespace square_ge_of_ge_pos_l3236_323620

theorem square_ge_of_ge_pos {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : a^2 ≥ b^2 := by
  sorry

end square_ge_of_ge_pos_l3236_323620


namespace orange_business_profit_l3236_323640

/-- Represents the profit calculation for Mr. Smith's orange business --/
theorem orange_business_profit :
  let small_oranges : ℕ := 5
  let medium_oranges : ℕ := 3
  let large_oranges : ℕ := 3
  let small_buy_price : ℚ := 1
  let medium_buy_price : ℚ := 2
  let large_buy_price : ℚ := 3
  let small_sell_price : ℚ := 1.5
  let medium_sell_price : ℚ := 3
  let large_sell_price : ℚ := 4
  let transportation_cost : ℚ := 2
  let storage_fee : ℚ := 1
  
  let total_buy_cost : ℚ := 
    small_oranges * small_buy_price + 
    medium_oranges * medium_buy_price + 
    large_oranges * large_buy_price +
    transportation_cost + storage_fee
  
  let total_sell_revenue : ℚ :=
    small_oranges * small_sell_price +
    medium_oranges * medium_sell_price +
    large_oranges * large_sell_price
  
  let profit : ℚ := total_sell_revenue - total_buy_cost
  
  profit = 5.5 := by sorry

end orange_business_profit_l3236_323640


namespace expression_factorization_l3236_323650

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 45 * x^2 - 10) - (-5 * x^3 + 15 * x^2 - 5) = 5 * (5 * x^3 + 6 * x^2 - 1) := by
  sorry

end expression_factorization_l3236_323650


namespace water_difference_proof_l3236_323694

/-- The difference in initial water amounts between Ji-hoon and Hyo-joo, given the conditions of the problem -/
def water_difference (j h : ℕ) : Prop :=
  (j - 152 = h + 152 + 346) → (j - h = 650)

/-- Theorem stating the water difference problem -/
theorem water_difference_proof :
  ∀ j h : ℕ, water_difference j h :=
by
  sorry

end water_difference_proof_l3236_323694


namespace bus_stop_time_l3236_323616

theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 50 →
  speed_with_stops = 35 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 18 := by
  sorry

end bus_stop_time_l3236_323616


namespace jim_reading_speed_increase_l3236_323600

-- Define Jim's reading parameters
def original_rate : ℝ := 40 -- pages per hour
def original_total : ℝ := 600 -- pages per week
def time_reduction : ℝ := 4 -- hours
def new_total : ℝ := 660 -- pages per week

-- Theorem statement
theorem jim_reading_speed_increase :
  let original_time := original_total / original_rate
  let new_time := original_time - time_reduction
  let new_rate := new_total / new_time
  new_rate / original_rate = 1.5
  := by sorry

end jim_reading_speed_increase_l3236_323600


namespace min_value_sqrt_sum_l3236_323631

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 1)^2 + (x + 2)^2) ≥ Real.sqrt 17 := by
  sorry

end min_value_sqrt_sum_l3236_323631


namespace total_stars_is_10_pow_22_l3236_323619

/-- The number of galaxies in the universe -/
def num_galaxies : ℕ := 10^11

/-- The number of stars in each galaxy -/
def stars_per_galaxy : ℕ := 10^11

/-- The total number of stars in the universe -/
def total_stars : ℕ := num_galaxies * stars_per_galaxy

/-- Theorem stating that the total number of stars is 10^22 -/
theorem total_stars_is_10_pow_22 : total_stars = 10^22 := by
  sorry

end total_stars_is_10_pow_22_l3236_323619


namespace monkey_doll_difference_l3236_323693

theorem monkey_doll_difference (total_budget : ℕ) (large_doll_cost : ℕ) (cost_difference : ℕ) : 
  total_budget = 300 → 
  large_doll_cost = 6 → 
  cost_difference = 2 → 
  (total_budget / (large_doll_cost - cost_difference) : ℕ) - (total_budget / large_doll_cost : ℕ) = 25 := by
  sorry

end monkey_doll_difference_l3236_323693


namespace range_of_a_l3236_323636

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 - x + 1/(16*a)

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l3236_323636


namespace brian_bought_22_pencils_l3236_323687

/-- The number of pencils Brian bought -/
def pencils_bought (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem stating that Brian bought 22 pencils -/
theorem brian_bought_22_pencils :
  pencils_bought 39 18 43 = 22 := by
  sorry

end brian_bought_22_pencils_l3236_323687


namespace star_seven_three_l3236_323637

/-- Custom binary operation ∗ -/
def star (a b : ℤ) : ℤ := 4*a + 5*b - a*b

/-- Theorem stating that 7 ∗ 3 = 22 -/
theorem star_seven_three : star 7 3 = 22 := by
  sorry

end star_seven_three_l3236_323637


namespace parabola_vertex_l3236_323671

/-- The parabola defined by y = (x-2)^2 + 4 has vertex at (2,4) -/
theorem parabola_vertex (x y : ℝ) :
  y = (x - 2)^2 + 4 → (2, 4) = (x, y) := by
  sorry

end parabola_vertex_l3236_323671


namespace arithmetic_sequence_sum_property_l3236_323606

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₁₀ = 16, then a₄ + a₈ = 16 -/
theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 2 + a 10 = 16) : 
  a 4 + a 8 = 16 := by
sorry

end arithmetic_sequence_sum_property_l3236_323606


namespace total_amount_paid_l3236_323664

def grape_quantity : ℕ := 3
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid : 
  grape_quantity * grape_rate + mango_quantity * mango_rate = 705 := by
  sorry

end total_amount_paid_l3236_323664


namespace pipe_filling_time_l3236_323675

theorem pipe_filling_time (fill_time_A : ℝ) (fill_time_B : ℝ) (combined_time : ℝ) :
  (fill_time_B = fill_time_A / 6) →
  (combined_time = 3.75) →
  (1 / fill_time_A + 1 / fill_time_B = 1 / combined_time) →
  fill_time_A = 26.25 := by
  sorry

end pipe_filling_time_l3236_323675


namespace arithmetic_progression_sum_l3236_323676

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  (a + 10 * d = 5.25) →
  (a + 6 * d = 3.25) →
  (n : ℝ) * (2 * a + (n - 1) * d) / 2 = 56.25 →
  n = 15 := by sorry

end arithmetic_progression_sum_l3236_323676


namespace lines_parallel_in_intersecting_planes_l3236_323695

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- State the theorem
theorem lines_parallel_in_intersecting_planes
  (l m n : Line) (α β γ : Plane)
  (distinct_lines : l ≠ m ∧ m ≠ n ∧ n ≠ l)
  (distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
  (h1 : intersect α β = l)
  (h2 : intersect β γ = m)
  (h3 : intersect γ α = n)
  (h4 : lineParallelPlane l γ) :
  parallel m n :=
sorry

end lines_parallel_in_intersecting_planes_l3236_323695


namespace extra_apples_l3236_323659

-- Define the number of red apples
def red_apples : ℕ := 6

-- Define the number of green apples
def green_apples : ℕ := 15

-- Define the number of students who wanted fruit
def students_wanting_fruit : ℕ := 5

-- Define the number of apples each student takes
def apples_per_student : ℕ := 1

-- Theorem to prove
theorem extra_apples : 
  (red_apples + green_apples) - (students_wanting_fruit * apples_per_student) = 16 := by
  sorry

end extra_apples_l3236_323659


namespace perpendicular_vectors_x_value_l3236_323632

/-- Given two vectors a and b in R², where a = (-2, 2) and b = (x, -3),
    if a is perpendicular to b, then x = -3. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![-2, 2]
  let b : Fin 2 → ℝ := ![x, -3]
  (∀ (i : Fin 2), a i * b i = 0) → x = -3 := by
  sorry

end perpendicular_vectors_x_value_l3236_323632


namespace teacher_distribution_l3236_323666

/-- The number of ways to distribute 4 teachers to 3 places -/
def distribute_teachers : ℕ := 36

/-- The number of ways to choose 2 teachers out of 4 -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 groups into 3 places -/
def arrange_three_groups : ℕ := 6

theorem teacher_distribution :
  distribute_teachers = choose_two_from_four * arrange_three_groups :=
sorry

end teacher_distribution_l3236_323666


namespace sandra_brought_twenty_pairs_l3236_323670

/-- Calculates the number of sock pairs Sandra brought given the initial conditions --/
def sandras_socks (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  let moms_pairs := 3 * initial_pairs + 8
  let s := (final_pairs - initial_pairs - moms_pairs) * 5 / 6
  s

theorem sandra_brought_twenty_pairs :
  sandras_socks 12 80 = 20 := by
  sorry

end sandra_brought_twenty_pairs_l3236_323670


namespace monkey_climb_theorem_l3236_323667

/-- The height of the pole in meters -/
def pole_height : ℝ := 10

/-- The time taken to reach the top of the pole in minutes -/
def total_time : ℕ := 17

/-- The distance the monkey slips in alternate minutes -/
def slip_distance : ℝ := 1

/-- The distance the monkey ascends in the first minute -/
def ascend_distance : ℝ := 1.8

/-- The number of complete ascend-slip cycles -/
def num_cycles : ℕ := (total_time - 1) / 2

theorem monkey_climb_theorem :
  ascend_distance + num_cycles * (ascend_distance - slip_distance) + ascend_distance = pole_height :=
sorry

end monkey_climb_theorem_l3236_323667


namespace nina_spiders_count_l3236_323630

/-- Proves that Nina has 3 spiders given the conditions of the problem -/
theorem nina_spiders_count :
  ∀ (spiders : ℕ),
  (∃ (total_eyes : ℕ),
    total_eyes = 124 ∧
    total_eyes = 8 * spiders + 2 * 50) →
  spiders = 3 := by
sorry

end nina_spiders_count_l3236_323630


namespace geometric_sequence_product_l3236_323611

/-- Given a geometric sequence {a_n} with a_1 = 1 and a_5 = 1/9, 
    prove that the product a_2 * a_3 * a_4 = 1/27 -/
theorem geometric_sequence_product (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                                -- first term
  a 5 = 1 / 9 →                            -- fifth term
  a 2 * a 3 * a 4 = 1 / 27 := by
sorry

end geometric_sequence_product_l3236_323611


namespace power_of_one_fourth_l3236_323661

theorem power_of_one_fourth (n : ℤ) : 1024 * (1 / 4 : ℚ) ^ n = 64 → n = 2 := by
  sorry

end power_of_one_fourth_l3236_323661


namespace maximum_value_inequality_l3236_323627

theorem maximum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (2 : ℝ) / 5 ≤ z) (h2 : z ≤ min x y) (h3 : x * z ≥ (4 : ℝ) / 15) (h4 : y * z ≥ (1 : ℝ) / 5) :
  (1 : ℝ) / x + 2 / y + 3 / z ≤ 13 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (2 : ℝ) / 5 ≤ z₀ ∧ z₀ ≤ min x₀ y₀ ∧ x₀ * z₀ ≥ (4 : ℝ) / 15 ∧ y₀ * z₀ ≥ (1 : ℝ) / 5 ∧
    (1 : ℝ) / x₀ + 2 / y₀ + 3 / z₀ = 13 := by
  sorry

end maximum_value_inequality_l3236_323627


namespace factorization_mn_minus_mn_cubed_l3236_323662

theorem factorization_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n^3 = m * n * (1 + n) * (1 - n) := by sorry

end factorization_mn_minus_mn_cubed_l3236_323662


namespace article_cost_l3236_323654

/-- The cost of an article given selling conditions. -/
theorem article_cost (sell_price_1 sell_price_2 : ℚ) (gain_increase : ℚ) : 
  sell_price_1 = 700 →
  sell_price_2 = 750 →
  gain_increase = 1/10 →
  ∃ (cost gain : ℚ), 
    cost + gain = sell_price_1 ∧
    cost + gain * (1 + gain_increase) = sell_price_2 ∧
    cost = 200 :=
by sorry

end article_cost_l3236_323654


namespace no_extreme_points_l3236_323685

/-- The function f(x) = x^3 - 3x^2 + 3x has no extreme points. -/
theorem no_extreme_points (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^3 - 3*x^2 + 3*x
  (∀ a b, a < b → f a < f b) :=
by
  sorry

end no_extreme_points_l3236_323685


namespace lilys_balance_proof_l3236_323602

/-- Calculates Lily's final account balance after a series of transactions --/
def lilys_final_balance (initial_amount shirt_cost book_price book_discount 
  savings_rate gift_percentage : ℚ) (num_books : ℕ) : ℚ :=
  let shoes_cost := 3 * shirt_cost
  let discounted_book_price := book_price * (1 - book_discount)
  let total_book_cost := (num_books : ℚ) * discounted_book_price
  let remaining_after_purchases := initial_amount - shirt_cost - shoes_cost - total_book_cost
  let savings := remaining_after_purchases / 2
  let savings_with_interest := savings * (1 + savings_rate)
  let gift_cost := savings_with_interest * gift_percentage
  savings_with_interest - gift_cost

theorem lilys_balance_proof :
  lilys_final_balance 55 7 8 0.2 0.2 0.25 4 = 0.63 := by
  sorry

end lilys_balance_proof_l3236_323602


namespace loot_box_average_loss_l3236_323613

/-- Represents the loot box problem with given parameters --/
structure LootBoxProblem where
  cost_per_box : ℝ
  standard_item_value : ℝ
  rare_item_value : ℝ
  rare_item_probability : ℝ
  total_spent : ℝ

/-- Calculates the average loss per loot box --/
def average_loss (p : LootBoxProblem) : ℝ :=
  let standard_prob := 1 - p.rare_item_probability
  let expected_value := standard_prob * p.standard_item_value + p.rare_item_probability * p.rare_item_value
  p.cost_per_box - expected_value

/-- Theorem stating the average loss per loot box --/
theorem loot_box_average_loss :
  let p : LootBoxProblem := {
    cost_per_box := 5,
    standard_item_value := 3.5,
    rare_item_value := 15,
    rare_item_probability := 0.1,
    total_spent := 40
  }
  average_loss p = 0.35 := by sorry

end loot_box_average_loss_l3236_323613


namespace f_properties_l3236_323635

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

def is_interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_range (f : ℝ → ℝ) (S : Set ℝ) (R : Set ℝ) : Prop :=
  ∀ y ∈ R, ∃ x ∈ S, f x = y

theorem f_properties (ω : ℝ) (h : ω > 0) :
  (∀ k : ℤ, is_interval_of_increase (f 1) (-Real.pi/3 + k*Real.pi) (Real.pi/6 + k*Real.pi)) ∧
  (ω = 8/3 → is_range (f ω) (Set.Icc 0 (Real.pi/8)) (Set.Icc (1/2) 1)) :=
sorry

end f_properties_l3236_323635


namespace sum_of_max_and_min_g_l3236_323610

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 10| + |x - 1|

theorem sum_of_max_and_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → g x ≤ max) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → min ≤ g x) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = min) ∧
    max + min = 3 :=
by sorry

end sum_of_max_and_min_g_l3236_323610


namespace complement_intersection_S_T_l3236_323638

def S : Finset Int := {-2, -1, 0, 1, 2}
def T : Finset Int := {-1, 0, 1}

theorem complement_intersection_S_T :
  (S \ (S ∩ T)) = {-2, 2} := by sorry

end complement_intersection_S_T_l3236_323638


namespace z₂_value_l3236_323679

-- Define the complex numbers
variable (z₁ z₂ : ℂ)

-- Define the conditions
axiom h₁ : (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
axiom h₂ : z₂.im = 2
axiom h₃ : (z₁ * z₂).im = 0

-- Theorem statement
theorem z₂_value : z₂ = 4 + 2 * Complex.I := by sorry

end z₂_value_l3236_323679


namespace factor_expression_l3236_323607

theorem factor_expression (b : ℝ) : 294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) := by
  sorry

end factor_expression_l3236_323607


namespace compute_F_2_f_3_l3236_323645

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 3*a + 2

-- Define function F
def F (a b : ℝ) : ℝ := b + a^3

-- Theorem to prove
theorem compute_F_2_f_3 : F 2 (f 3) = 10 := by
  sorry

end compute_F_2_f_3_l3236_323645


namespace complex_magnitude_problem_l3236_323689

theorem complex_magnitude_problem (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end complex_magnitude_problem_l3236_323689


namespace regular_polygon_18_degree_exterior_angles_l3236_323609

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_18_degree_exterior_angles (n : ℕ) : 
  (n > 0) → (360 / n = 18) → n = 20 := by sorry

end regular_polygon_18_degree_exterior_angles_l3236_323609


namespace polynomial_factorization_l3236_323668

theorem polynomial_factorization (a b : ℤ) :
  (∀ x : ℝ, 24 * x^2 - 158 * x - 147 = (12 * x + a) * (2 * x + b)) →
  a + 2 * b = -35 := by
  sorry

end polynomial_factorization_l3236_323668


namespace quadratic_equation_condition_l3236_323623

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 :=
by sorry

end quadratic_equation_condition_l3236_323623


namespace problem_pyramid_volume_l3236_323673

/-- Triangular pyramid with given side lengths -/
structure TriangularPyramid where
  base_side : ℝ
  pa : ℝ
  pb : ℝ
  pc : ℝ

/-- Volume of a triangular pyramid -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- The specific triangular pyramid from the problem -/
def problem_pyramid : TriangularPyramid :=
  { base_side := 3
  , pa := 3
  , pb := 4
  , pc := 5 }

/-- Theorem stating that the volume of the problem pyramid is √11 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = Real.sqrt 11 := by
  sorry

end problem_pyramid_volume_l3236_323673


namespace negation_of_existence_squared_greater_than_power_of_two_l3236_323639

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_existence_squared_greater_than_power_of_two_l3236_323639


namespace unique_four_digit_number_l3236_323618

theorem unique_four_digit_number :
  ∃! N : ℕ,
    N ≡ N^2 [ZMOD 10000] ∧
    N ≡ 7 [ZMOD 16] ∧
    1000 ≤ N ∧ N < 10000 ∧
    N = 3751 := by sorry

end unique_four_digit_number_l3236_323618


namespace max_time_sum_of_digits_l3236_323621

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : ℕ := 24

theorem max_time_sum_of_digits :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits := by sorry

end max_time_sum_of_digits_l3236_323621


namespace acute_triangle_inequality_l3236_323624

/-- Triangle with acute angle opposite to side c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angle : c^2 < a^2 + b^2

theorem acute_triangle_inequality (t : AcuteTriangle) :
  (t.a^2 + t.b^2 + t.c^2) / (t.a^2 + t.b^2) > 1 :=
sorry

end acute_triangle_inequality_l3236_323624


namespace product_of_nonneg_quadratics_is_nonneg_l3236_323677

/-- Given two non-negative quadratic functions, their product is also non-negative. -/
theorem product_of_nonneg_quadratics_is_nonneg
  (a b c A B C : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0)
  (h2 : ∀ x : ℝ, A * x^2 + 2 * B * x + C ≥ 0) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by sorry

end product_of_nonneg_quadratics_is_nonneg_l3236_323677


namespace y_coord_at_neg_three_l3236_323692

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_value : ℝ
  max_x : ℝ
  point_zero : ℝ
  has_max : max_value = 7
  max_at : max_x = -2
  passes_zero : a * 0^2 + b * 0 + c = point_zero
  passes_zero_value : point_zero = -15

/-- The y-coordinate of a point on the quadratic function -/
def y_coord (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem stating the y-coordinate at x = -3 is 1.5 -/
theorem y_coord_at_neg_three (f : QuadraticFunction) : y_coord f (-3) = 1.5 := by
  sorry

end y_coord_at_neg_three_l3236_323692


namespace det_2x2_matrix_l3236_323691

theorem det_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 3]
  Matrix.det A = 2 := by
  sorry

end det_2x2_matrix_l3236_323691


namespace initial_students_per_class_l3236_323622

theorem initial_students_per_class 
  (initial_classes : ℕ) 
  (added_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : initial_classes = 15)
  (h2 : added_classes = 5)
  (h3 : total_students = 400) :
  (total_students / (initial_classes + added_classes) : ℚ) = 20 := by
sorry

end initial_students_per_class_l3236_323622


namespace two_colonies_limit_days_l3236_323647

/-- Represents the number of days it takes for a single bacteria colony to reach the habitat limit -/
def single_colony_limit_days : ℕ := 20

/-- Represents the growth rate of the bacteria colony (doubling every day) -/
def growth_rate : ℚ := 2

/-- Represents the fixed habitat limit -/
def habitat_limit : ℚ := growth_rate ^ single_colony_limit_days

/-- Theorem stating that two colonies reach the habitat limit in the same number of days as one colony -/
theorem two_colonies_limit_days (initial_colonies : ℕ) (h : initial_colonies = 2) :
  (initial_colonies * growth_rate ^ single_colony_limit_days = habitat_limit) :=
sorry

end two_colonies_limit_days_l3236_323647


namespace fraction_value_l3236_323684

theorem fraction_value (N : ℝ) (h : 0.4 * N = 168) : (1/4) * (1/3) * (2/5) * N = 14 := by
  sorry

end fraction_value_l3236_323684


namespace sum_of_fractions_l3236_323698

theorem sum_of_fractions : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = 1 / 2 := by
  sorry

end sum_of_fractions_l3236_323698
