import Mathlib

namespace production_rate_equation_l1744_174459

/-- Represents the production rates of a master and apprentice -/
theorem production_rate_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 40) 
  (h3 : x + (40 - x) = 40) 
  (h4 : ∃ t : ℝ, t > 0 ∧ x * t = 300 ∧ (40 - x) * t = 100) : 
  300 / x = 100 / (40 - x) := by
  sorry

end production_rate_equation_l1744_174459


namespace remainder_problem_l1744_174419

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) : k % 7 = 3 := by
  sorry

end remainder_problem_l1744_174419


namespace base5_to_base7_conversion_l1744_174432

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

theorem base5_to_base7_conversion :
  decimalToBase7 (base5ToDecimal 412) = 212 := by sorry

end base5_to_base7_conversion_l1744_174432


namespace light_bulbs_not_broken_l1744_174415

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def num_not_broken (kitchen_total : ℕ) (foyer_broken : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_total) / 5
  let kitchen_not_broken := kitchen_total - kitchen_broken
  let foyer_total := foyer_broken * 3
  let foyer_not_broken := foyer_total - foyer_broken
  kitchen_not_broken + foyer_not_broken

/-- Theorem stating that the number of light bulbs not broken in both the foyer and kitchen is 34 -/
theorem light_bulbs_not_broken :
  num_not_broken 35 10 = 34 := by
  sorry

end light_bulbs_not_broken_l1744_174415


namespace five_boys_three_girls_arrangements_l1744_174447

/-- The number of arrangements of boys and girls in a row -/
def arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial (num_boys + 1)) * (Nat.factorial num_girls)

/-- Theorem stating the number of arrangements for 5 boys and 3 girls -/
theorem five_boys_three_girls_arrangements :
  arrangements 5 3 = 4320 := by
  sorry

end five_boys_three_girls_arrangements_l1744_174447


namespace card_value_decrease_l1744_174487

theorem card_value_decrease (x : ℝ) : 
  (1 - x/100) * (1 - x/100) = 0.81 → x = 10 := by
sorry

end card_value_decrease_l1744_174487


namespace log_equality_l1744_174426

theorem log_equality (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 5 = (1/2) * (Real.log 2 / Real.log 5) := by
  sorry

end log_equality_l1744_174426


namespace vanessa_score_l1744_174445

/-- Vanessa's basketball game score calculation -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (other_avg : ℕ) : 
  total_score = 65 → other_players = 7 → other_avg = 5 → 
  total_score - (other_players * other_avg) = 30 := by
sorry

end vanessa_score_l1744_174445


namespace principal_amount_calculation_l1744_174453

theorem principal_amount_calculation (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 0.08333333333333334 →
  interest = 400 →
  time = 4 →
  interest = (interest / (rate * time)) * rate * time :=
by
  sorry

end principal_amount_calculation_l1744_174453


namespace cube_root_of_three_times_two_to_fifth_l1744_174450

theorem cube_root_of_three_times_two_to_fifth (x : ℝ) : 
  x^3 = 2^5 + 2^5 + 2^5 → x = 6 * 6^(2/3) :=
by sorry

end cube_root_of_three_times_two_to_fifth_l1744_174450


namespace article_pricing_gain_percent_l1744_174462

/-- Proves that if the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is 300/7. -/
theorem article_pricing_gain_percent
  (C : ℝ) -- Cost price of one article
  (S : ℝ) -- Selling price of one article
  (h : 50 * C = 35 * S) -- Given condition
  : (S - C) / C * 100 = 300 / 7 := by
  sorry

end article_pricing_gain_percent_l1744_174462


namespace frustum_radius_l1744_174400

theorem frustum_radius (r : ℝ) :
  (r > 0) →
  (3 * (2 * π * r) = 2 * π * (3 * r)) →
  (π * (r + 3 * r) * 3 = 84 * π) →
  r = 7 := by sorry

end frustum_radius_l1744_174400


namespace special_integers_proof_l1744_174455

theorem special_integers_proof (k : ℕ) (h : k ≥ 2) :
  (∀ m n : ℕ, 1 ≤ m ∧ m < n ∧ n ≤ k → ¬(k ∣ (n^(n-1) - m^(m-1)))) ↔ (k = 2 ∨ k = 3) :=
sorry

end special_integers_proof_l1744_174455


namespace cube_volume_from_surface_area_l1744_174471

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end cube_volume_from_surface_area_l1744_174471


namespace square_divisibility_l1744_174405

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem square_divisibility (x y : ℕ) : 
  x > 0 → y > 0 → x > y → divides (x * y) (x^2022 + x + y^2) → ∃ n : ℕ, x = n^2 := by
  sorry

end square_divisibility_l1744_174405


namespace complex_cube_root_unity_l1744_174482

/-- Given that i is the imaginary unit and z = -1/2 + (√3/2)i, prove that z^2 + z + 1 = 0 -/
theorem complex_cube_root_unity (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = -1/2 + (Real.sqrt 3 / 2) * i →
  z^2 + z + 1 = 0 := by
  sorry

end complex_cube_root_unity_l1744_174482


namespace triangle_angle_difference_l1744_174484

/-- In a triangle with angles a, b, and c, where b = 2a and c = a - 15, 
    prove that a - c = 15 --/
theorem triangle_angle_difference (a b c : ℝ) : 
  a + b + c = 180 → b = 2 * a → c = a - 15 → a - c = 15 := by
  sorry

end triangle_angle_difference_l1744_174484


namespace min_cost_14000_l1744_174468

/-- Represents the number of soup + main course combinations -/
def x : ℕ := 15

/-- Represents the number of salad + main course combinations -/
def y : ℕ := 0

/-- Represents the number of all three dish combinations -/
def z : ℕ := 0

/-- Represents the number of standalone main courses -/
def q : ℕ := 35

/-- The cost of a salad -/
def salad_cost : ℕ := 200

/-- The cost of soup + main course -/
def soup_main_cost : ℕ := 350

/-- The cost of salad + main course -/
def salad_main_cost : ℕ := 350

/-- The cost of soup + salad + main course -/
def all_three_cost : ℕ := 500

/-- The total number of main courses required -/
def total_main : ℕ := 50

/-- The total number of salads required -/
def total_salad : ℕ := 30

/-- The total number of soups required -/
def total_soup : ℕ := 15

theorem min_cost_14000 :
  (x + y + z + q = total_main) ∧
  (y + z = total_salad) ∧
  (x + z = total_soup) ∧
  (∀ x' y' z' q' : ℕ,
    (x' + y' + z' + q' = total_main) →
    (y' + z' = total_salad) →
    (x' + z' = total_soup) →
    soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q ≤
    soup_main_cost * x' + salad_main_cost * y' + all_three_cost * z' + salad_cost * q') →
  soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q = 14000 :=
sorry

end min_cost_14000_l1744_174468


namespace complex_number_in_fourth_quadrant_l1744_174444

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 4 / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end complex_number_in_fourth_quadrant_l1744_174444


namespace vanilla_cookie_price_l1744_174440

theorem vanilla_cookie_price 
  (chocolate_count : ℕ) 
  (vanilla_count : ℕ) 
  (chocolate_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : chocolate_price = 1)
  (h4 : total_revenue = 360) :
  ∃ (vanilla_price : ℚ), 
    vanilla_price = 2 ∧ 
    chocolate_count * chocolate_price + vanilla_count * vanilla_price = total_revenue :=
by sorry

end vanilla_cookie_price_l1744_174440


namespace line_plane_intersection_equivalence_l1744_174485

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Line → Line → Prop)
variable (within : Line → Plane → Prop)
variable (intersects_plane : Line → Plane → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line)
variable (α β : Plane)

-- State the theorem
theorem line_plane_intersection_equivalence 
  (h1 : intersect l m)
  (h2 : within l α)
  (h3 : within m α)
  (h4 : ¬ within l β)
  (h5 : ¬ within m β) :
  (intersects_plane l β ∨ intersects_plane m β) ↔ planes_intersect α β := by
  sorry

end line_plane_intersection_equivalence_l1744_174485


namespace partial_fraction_decomposition_l1744_174461

theorem partial_fraction_decomposition (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℚ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = r / (a * x + b) + s / (c * x + d) ∧
    r = a / (a * d - b * c) ∧
    s = -c / (a * d - b * c) := by
  sorry

end partial_fraction_decomposition_l1744_174461


namespace second_number_problem_l1744_174457

theorem second_number_problem (A B C : ℝ) (h_sum : A + B + C = 98) 
  (h_ratio1 : A / B = 2 / 3) (h_ratio2 : B / C = 5 / 8) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  B = 30 := by
sorry

end second_number_problem_l1744_174457


namespace rectangle_diagonal_squares_l1744_174448

/-- The number of unit squares that the diagonals of a rectangle pass through -/
def diagonalSquares (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width - 1 + height - 1 + 1) - 2

/-- Theorem: For a 20 × 19 rectangle with one corner at the origin and sides parallel to the coordinate axes,
    the number of unit squares that the two diagonals pass through is 74. -/
theorem rectangle_diagonal_squares :
  diagonalSquares 20 19 = 74 := by
  sorry

end rectangle_diagonal_squares_l1744_174448


namespace assembly_line_arrangements_l1744_174466

theorem assembly_line_arrangements (n : ℕ) (arrangements : ℕ) 
  (h1 : n = 6) 
  (h2 : arrangements = 360) :
  arrangements = n.factorial / 2 := by
sorry

end assembly_line_arrangements_l1744_174466


namespace diamond_3_7_l1744_174411

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := star a b - a*b

-- Theorem to prove
theorem diamond_3_7 : diamond 3 7 = 79 := by
  sorry

end diamond_3_7_l1744_174411


namespace range_of_k_for_fractional_equation_l1744_174408

theorem range_of_k_for_fractional_equation :
  ∀ k x : ℝ,
  (x > 0) →
  (x ≠ 2) →
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) →
  (k > -2 ∧ k ≠ 4) :=
by sorry

end range_of_k_for_fractional_equation_l1744_174408


namespace trig_identity_proof_l1744_174403

theorem trig_identity_proof : 
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_proof_l1744_174403


namespace point_comparison_l1744_174479

/-- Given that points (-2, y₁) and (-1, y₂) lie on the line y = -3x + b, prove that y₁ > y₂ -/
theorem point_comparison (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -3 * (-2) + b) 
  (h₂ : y₂ = -3 * (-1) + b) : 
  y₁ > y₂ := by
  sorry


end point_comparison_l1744_174479


namespace relationship_abc_l1744_174475

theorem relationship_abc : ∀ a b c : ℝ,
  a = -1/3 * 9 →
  b = 2 - 4 →
  c = 2 / (-1/2) →
  c < a ∧ a < b :=
by
  sorry

end relationship_abc_l1744_174475


namespace sum_of_distances_is_12_sqrt_2_l1744_174460

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def P : ℝ × ℝ := (-2, -4)

-- Define the intersection points M and N (existence assumed)
axiom M_exists : ∃ M : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2
axiom N_exists : ∃ N : ℝ × ℝ, C N.1 N.2 ∧ l N.1 N.2
axiom M_ne_N : ∀ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 → M ≠ N

-- Theorem statement
theorem sum_of_distances_is_12_sqrt_2 :
  ∃ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 ∧ M ≠ N ∧
  Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) + Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 12 * Real.sqrt 2 :=
sorry

end sum_of_distances_is_12_sqrt_2_l1744_174460


namespace probability_three_different_suits_l1744_174451

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
theorem probability_three_different_suits : 
  (39 : ℚ) / 51 * 24 / 50 = 156 / 425 := by
  sorry

end probability_three_different_suits_l1744_174451


namespace residue_of_7_500_mod_19_l1744_174465

theorem residue_of_7_500_mod_19 : 7^500 % 19 = 15 := by
  sorry

end residue_of_7_500_mod_19_l1744_174465


namespace pria_distance_driven_l1744_174424

/-- Calculates the distance driven with a full tank of gas given the advertised mileage,
    tank capacity, and difference between advertised and actual mileage. -/
def distance_driven (advertised_mileage : ℝ) (tank_capacity : ℝ) (mileage_difference : ℝ) : ℝ :=
  (advertised_mileage - mileage_difference) * tank_capacity

/-- Proves that given the specified conditions, the distance driven is 372 miles. -/
theorem pria_distance_driven :
  distance_driven 35 12 4 = 372 := by
  sorry

end pria_distance_driven_l1744_174424


namespace isolation_process_complete_l1744_174483

/-- Represents a step in the process of isolating and counting bacteria --/
inductive ProcessStep
  | SoilSampling
  | SampleDilution
  | SpreadingDilution
  | SelectingColonies
  | Identification

/-- Represents the process of isolating and counting bacteria that decompose urea in soil --/
def IsolationProcess : List ProcessStep := 
  [ProcessStep.SoilSampling, 
   ProcessStep.SampleDilution, 
   ProcessStep.SpreadingDilution, 
   ProcessStep.SelectingColonies, 
   ProcessStep.Identification]

/-- The theorem states that the IsolationProcess contains all necessary steps in the correct order --/
theorem isolation_process_complete : 
  IsolationProcess = 
    [ProcessStep.SoilSampling, 
     ProcessStep.SampleDilution, 
     ProcessStep.SpreadingDilution, 
     ProcessStep.SelectingColonies, 
     ProcessStep.Identification] := by
  sorry


end isolation_process_complete_l1744_174483


namespace crackers_per_friend_l1744_174456

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end crackers_per_friend_l1744_174456


namespace hotel_stay_cost_l1744_174428

/-- Calculates the total cost for a group staying at a hotel. -/
def total_hotel_cost (cost_per_night : ℕ) (num_nights : ℕ) (num_people : ℕ) : ℕ :=
  cost_per_night * num_nights * num_people

/-- Proves that the total cost for 3 people staying 3 nights at $40 per night is $360. -/
theorem hotel_stay_cost :
  total_hotel_cost 40 3 3 = 360 := by
sorry

end hotel_stay_cost_l1744_174428


namespace bens_age_l1744_174474

theorem bens_age (b j : ℕ) : 
  b = 3 * j + 10 →  -- Ben's age is 10 years more than thrice Jane's age
  b + j = 70 →      -- The sum of their ages is 70
  b = 55 :=         -- Ben's age is 55
by sorry

end bens_age_l1744_174474


namespace salt_solution_percentage_l1744_174427

def is_valid_salt_solution (initial_salt_percent : ℝ) : Prop :=
  let replaced_volume : ℝ := 1/4
  let final_salt_percent : ℝ := 16
  let replacing_salt_percent : ℝ := 31
  (1 - replaced_volume) * initial_salt_percent + replaced_volume * replacing_salt_percent = final_salt_percent

theorem salt_solution_percentage :
  ∃ (x : ℝ), is_valid_salt_solution x ∧ x = 11 :=
sorry

end salt_solution_percentage_l1744_174427


namespace triangle_inequality_l1744_174481

theorem triangle_inequality (a b c : ℝ) (h : |((a^2 + b^2 - c^2) / (a*b))| < 2) :
  |((b^2 + c^2 - a^2) / (b*c))| < 2 ∧ |((c^2 + a^2 - b^2) / (c*a))| < 2 := by
  sorry

end triangle_inequality_l1744_174481


namespace even_function_sufficient_not_necessary_l1744_174410

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def exists_symmetric_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = f (-x₀)

theorem even_function_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, is_even_function f → exists_symmetric_point f) ∧
  ¬(∀ f : ℝ → ℝ, exists_symmetric_point f → is_even_function f) := by
  sorry

end even_function_sufficient_not_necessary_l1744_174410


namespace smoothie_cost_l1744_174473

/-- The cost of Morgan's smoothie given the prices of other items and the transaction details. -/
theorem smoothie_cost (hamburger_cost onion_rings_cost amount_paid change_received : ℕ) : 
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  amount_paid = 20 →
  change_received = 11 →
  amount_paid - change_received - (hamburger_cost + onion_rings_cost) = 3 := by
  sorry

#check smoothie_cost

end smoothie_cost_l1744_174473


namespace percentage_studying_both_languages_l1744_174446

def english_percentage : ℝ := 90
def german_percentage : ℝ := 80

theorem percentage_studying_both_languages :
  let both_percentage := english_percentage + german_percentage - 100
  both_percentage = 70 := by sorry

end percentage_studying_both_languages_l1744_174446


namespace cost_of_three_pencils_four_pens_l1744_174433

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The cost of 8 pencils and 3 pens is $5.60 -/
axiom first_equation : 8 * pencil_cost + 3 * pen_cost = 5.60

/-- The cost of 2 pencils and 5 pens is $4.25 -/
axiom second_equation : 2 * pencil_cost + 5 * pen_cost = 4.25

/-- The cost of 3 pencils and 4 pens is approximately $9.68 -/
theorem cost_of_three_pencils_four_pens :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |3 * pencil_cost + 4 * pen_cost - 9.68| < ε :=
sorry

end cost_of_three_pencils_four_pens_l1744_174433


namespace root_difference_l1744_174472

-- Define the equations
def equation1 (x : ℝ) : Prop := 2002^2 * x^2 - 2003 * 2001 * x - 1 = 0
def equation2 (x : ℝ) : Prop := 2001 * x^2 - 2002 * x + 1 = 0

-- Define r and s
def r : ℝ := sorry
def s : ℝ := sorry

-- State the theorem
theorem root_difference : 
  (equation1 r ∧ ∀ x, equation1 x → x ≤ r) ∧ 
  (equation2 s ∧ ∀ x, equation2 x → x ≥ s) → 
  r - s = 2000 / 2001 := by sorry

end root_difference_l1744_174472


namespace will_chocolate_boxes_l1744_174409

theorem will_chocolate_boxes :
  ∀ (boxes_given : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ),
    boxes_given = 3 →
    pieces_per_box = 4 →
    pieces_left = 16 →
    (boxes_given * pieces_per_box + pieces_left) / pieces_per_box = 7 :=
by
  sorry

end will_chocolate_boxes_l1744_174409


namespace wood_per_sack_l1744_174407

theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) (wood_per_sack : ℕ) 
  (h1 : total_wood = 80) 
  (h2 : num_sacks = 4) 
  (h3 : wood_per_sack = total_wood / num_sacks) :
  wood_per_sack = 20 := by
  sorry

end wood_per_sack_l1744_174407


namespace M_N_intersection_empty_l1744_174423

def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * ((1 + t) / t)}

def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

theorem M_N_intersection_empty : M ∩ N = ∅ := by
  sorry

end M_N_intersection_empty_l1744_174423


namespace plane_equation_l1744_174464

/-- The equation of a plane passing through a point and parallel to another plane -/
theorem plane_equation (x y z : ℝ) : ∃ (A B C D : ℤ),
  -- The plane passes through the point (2,3,-1)
  A * 2 + B * 3 + C * (-1) + D = 0 ∧
  -- The plane is parallel to 3x - 4y + 2z = 5
  ∃ (k : ℝ), k ≠ 0 ∧ A = k * 3 ∧ B = k * (-4) ∧ C = k * 2 ∧
  -- The equation is in the form Ax + By + Cz + D = 0
  A * x + B * y + C * z + D = 0 ∧
  -- A is positive
  A > 0 ∧
  -- The greatest common divisor of |A|, |B|, |C|, and |D| is 1
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  -- The specific solution
  A = 3 ∧ B = -4 ∧ C = 2 ∧ D = 8 := by
sorry

end plane_equation_l1744_174464


namespace triangulations_equal_catalan_l1744_174492

/-- Number of triangulations of an n-sided polygon -/
def T (n : ℕ) : ℕ := sorry

/-- Catalan numbers -/
def C (n : ℕ) : ℕ := sorry

/-- Theorem: The number of triangulations of an n-sided polygon
    is equal to the (n-2)th Catalan number -/
theorem triangulations_equal_catalan (n : ℕ) : T n = C (n - 2) := by sorry

end triangulations_equal_catalan_l1744_174492


namespace otimes_example_otimes_sum_property_l1744_174429

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem 1
theorem otimes_example : otimes 2 (-2) = 6 := by sorry

-- Theorem 2
theorem otimes_sum_property (a b : ℝ) (h : a + b = 0) : 
  (otimes a a) + (otimes b b) = 2 * a * b := by sorry

end otimes_example_otimes_sum_property_l1744_174429


namespace system_solution_l1744_174458

theorem system_solution (u v w : ℝ) : 
  (u + v * w = 20 ∧ v + w * u = 20 ∧ w + u * v = 20) ↔ 
  ((u, v, w) = (4, 4, 4) ∨ 
   (u, v, w) = (-5, -5, -5) ∨ 
   (u, v, w) = (1, 1, 19) ∨ 
   (u, v, w) = (19, 1, 1) ∨ 
   (u, v, w) = (1, 19, 1)) := by
sorry

end system_solution_l1744_174458


namespace square_sum_inequality_l1744_174430

theorem square_sum_inequality (x y : ℝ) : x^2 + y^2 + 1 ≥ x + y + x*y := by
  sorry

end square_sum_inequality_l1744_174430


namespace tina_career_difference_l1744_174431

def boxing_career (initial_wins : ℕ) (second_wins : ℕ) (third_wins : ℕ) (fourth_wins : ℕ) : ℕ := 
  let wins1 := initial_wins + second_wins
  let wins2 := wins1 * 3
  let wins3 := wins2 + third_wins
  let wins4 := wins3 * 2
  let wins5 := wins4 + fourth_wins
  wins5 * wins5

theorem tina_career_difference : 
  boxing_career 10 5 7 11 - 4 = 13221 := by sorry

end tina_career_difference_l1744_174431


namespace f_inequality_iff_a_bound_l1744_174498

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

theorem f_inequality_iff_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ 1 / (Real.exp 1 - 1) := by sorry

end f_inequality_iff_a_bound_l1744_174498


namespace intersection_complement_equals_l1744_174454

def U : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def M : Set ℕ := {1, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals : M ∩ (U \ N) = {1, 5} := by sorry

end intersection_complement_equals_l1744_174454


namespace lost_revenue_calculation_l1744_174443

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (capacity : ℕ) (general_price : ℚ) (child_price : ℚ) (senior_price : ℚ) 
  (veteran_discount : ℚ) (general_sold : ℕ) (child_sold : ℕ) (senior_sold : ℕ) (veteran_sold : ℕ) : ℚ :=
  let actual_revenue := general_sold * general_price + child_sold * child_price + 
                        senior_sold * senior_price + veteran_sold * (general_price - veteran_discount)
  let max_potential_revenue := capacity * general_price
  max_potential_revenue - actual_revenue

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation : 
  theater_revenue 50 10 6 8 2 20 3 4 2 = 234 := by sorry

end lost_revenue_calculation_l1744_174443


namespace simplify_expression_l1744_174490

theorem simplify_expression (x y : ℝ) : (5*x - y) - 3*(2*x - 3*y) + x = 8*y := by
  sorry

end simplify_expression_l1744_174490


namespace product_inequality_l1744_174480

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  let M := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)
  M ≤ -8 := by
  sorry

end product_inequality_l1744_174480


namespace orange_buckets_total_l1744_174416

theorem orange_buckets_total (x y : ℝ) : 
  x = 2 * 22.5 + 3 →
  y = x - 11.5 →
  22.5 + x + y = 107 := by
sorry

end orange_buckets_total_l1744_174416


namespace fifth_power_last_digit_l1744_174478

theorem fifth_power_last_digit (n : ℕ) : 10 ∣ (n^5 - n) := by
  sorry

end fifth_power_last_digit_l1744_174478


namespace num_solutions_is_four_l1744_174442

/-- The number of distinct solutions to the system of equations:
    (x - 2y + 3)(4x + y - 5) = 0 and (x + 2y - 5)(3x - 4y + 6) = 0 -/
def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := (x - 2*y + 3)*(4*x + y - 5) = 0
  let eq2 (x y : ℝ) := (x + 2*y - 5)*(3*x - 4*y + 6) = 0
  4  -- The actual number of solutions

theorem num_solutions_is_four :
  num_solutions = 4 := by sorry

end num_solutions_is_four_l1744_174442


namespace solution_value_l1744_174421

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the parameters a and b
variable (a b : ℝ)

-- State the theorem
theorem solution_value (h : {x | a*x^2 + b*x + 2 > 0} = A ∩ B) : a + b = -2 := by
  sorry

end solution_value_l1744_174421


namespace sum_of_square_areas_l1744_174467

theorem sum_of_square_areas : 
  let square1_side : ℝ := 8
  let square2_side : ℝ := 10
  let square1_area : ℝ := square1_side * square1_side
  let square2_area : ℝ := square2_side * square2_side
  square1_area + square2_area = 164 :=
by sorry

end sum_of_square_areas_l1744_174467


namespace annulus_area_l1744_174437

/-- An annulus is formed by two concentric circles with radii R and r, where R > r.
    x is the length of a tangent line from a point on the outer circle to the inner circle. -/
theorem annulus_area (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 := by sorry

end annulus_area_l1744_174437


namespace angle_A_is_pi_over_4_area_is_8_l1744_174404

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.sqrt 2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A

-- Theorem 1: Angle A is π/4
theorem angle_A_is_pi_over_4 (t : Triangle) (h : satisfiesCondition t) : t.A = π / 4 := by
  sorry

-- Theorem 2: Area of the triangle is 8 under specific conditions
theorem area_is_8 (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 10) (h3 : t.b = 8 * Real.sqrt 2) 
  (h4 : t.C < t.A ∧ t.C < t.B) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 8 := by
  sorry

end angle_A_is_pi_over_4_area_is_8_l1744_174404


namespace cost_dozen_pens_l1744_174493

-- Define the cost of one pen and one pencil
def cost_pen : ℝ := sorry
def cost_pencil : ℝ := sorry

-- Define the conditions
axiom total_cost : 3 * cost_pen + 5 * cost_pencil = 100
axiom cost_ratio : cost_pen = 5 * cost_pencil

-- Theorem to prove
theorem cost_dozen_pens : 12 * cost_pen = 300 := by
  sorry

end cost_dozen_pens_l1744_174493


namespace trigonometric_inequality_l1744_174463

theorem trigonometric_inequality (x : ℝ) :
  9.286 * (Real.sin x)^3 * Real.sin (π/2 - 3*x) + (Real.cos x)^3 * Real.cos (π/2 - 3*x) > 3*Real.sqrt 3/8 →
  ∃ n : ℤ, π/12 + n*π/2 < x ∧ x < π/6 + n*π/2 := by
sorry

end trigonometric_inequality_l1744_174463


namespace infinitely_many_primes_with_property_l1744_174491

-- Define the property for a prime p
def hasDivisibilityProperty (p : Nat) : Prop :=
  ∃ k : Nat, k > 0 ∧ p ∣ (2^k - 3)

-- State the theorem
theorem infinitely_many_primes_with_property :
  ∀ n : Nat, ∃ p : Nat, p > n ∧ Prime p ∧ hasDivisibilityProperty p := by
  sorry

end infinitely_many_primes_with_property_l1744_174491


namespace journey_fraction_l1744_174452

theorem journey_fraction (total_distance : ℝ) (bus_fraction : ℚ) (foot_distance : ℝ) :
  total_distance = 130 →
  bus_fraction = 17 / 20 →
  foot_distance = 6.5 →
  ∃ rail_fraction : ℚ,
    rail_fraction + bus_fraction + (foot_distance / total_distance) = 1 ∧
    rail_fraction = 1 / 10 :=
by sorry

end journey_fraction_l1744_174452


namespace f_extrema_l1744_174412

def f (x : ℝ) : ℝ := -x^3 + 3*x - 1

theorem f_extrema :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -3 ∧ ∀ y : ℝ, f y ≥ f x) :=
by sorry

end f_extrema_l1744_174412


namespace min_socks_for_ten_pairs_five_colors_l1744_174499

/-- The minimum number of socks needed to guarantee a certain number of pairs, given a number of colors -/
def min_socks (colors : ℕ) (pairs : ℕ) : ℕ := 2 * pairs + colors - 1

/-- Theorem stating that 24 socks are needed to guarantee 10 pairs with 5 colors -/
theorem min_socks_for_ten_pairs_five_colors :
  min_socks 5 10 = 24 := by sorry

end min_socks_for_ten_pairs_five_colors_l1744_174499


namespace right_triangle_existence_l1744_174434

theorem right_triangle_existence (c h : ℝ) (hc : c > 0) (hh : h > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (a * b) / c = h ↔ h ≤ c / 2 :=
sorry

end right_triangle_existence_l1744_174434


namespace geometric_series_remainder_remainder_of_series_l1744_174418

theorem geometric_series_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = 
  ((a * (r^n % (m * (r - 1))) - a) / (r - 1)) % m :=
sorry

theorem remainder_of_series : 
  (((3^1002 - 1) / 2) % 500) = 4 :=
sorry

end geometric_series_remainder_remainder_of_series_l1744_174418


namespace cake_muffin_probability_l1744_174477

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 16) :
  (total - (cake + muffin - both)) / total = 26 / 100 := by
sorry

end cake_muffin_probability_l1744_174477


namespace quadratic_perfect_square_l1744_174420

/-- If 9x^2 - 24x + c is a perfect square of a binomial, then c = 16 -/
theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end quadratic_perfect_square_l1744_174420


namespace solve_linear_equation_l1744_174401

theorem solve_linear_equation (x y : ℝ) :
  2 * x + y = 5 → x = (5 - y) / 2 := by
sorry

end solve_linear_equation_l1744_174401


namespace some_number_value_l1744_174494

theorem some_number_value : ∃ n : ℤ, (481 + 426) * n - 4 * 481 * 426 = 3025 ∧ n = 906 := by
  sorry

end some_number_value_l1744_174494


namespace not_divisible_by_59_l1744_174425

theorem not_divisible_by_59 (x y : ℕ) 
  (h1 : ¬ 59 ∣ x) 
  (h2 : ¬ 59 ∣ y) 
  (h3 : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end not_divisible_by_59_l1744_174425


namespace sequence_inequality_l1744_174470

def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def not_in_sequence (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ n : ℕ, a n ≠ x

def representable (a : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, not_in_sequence a x → ∃ k : ℕ, x = a k + 2 * k

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : is_increasing a) 
  (h2 : representable a) : 
  ∀ k : ℕ, (a k : ℝ) < Real.sqrt (2 * k) := by
  sorry

end sequence_inequality_l1744_174470


namespace right_triangle_circles_coincide_l1744_174449

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_B : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circles
def circle_BC (B C D : ℝ × ℝ) : Prop :=
  (D.1 - B.1) * (C.1 - D.1) + (D.2 - B.2) * (C.2 - D.2) = 0

def circle_AB (A B E : ℝ × ℝ) : Prop :=
  (E.1 - A.1) * (B.1 - E.1) + (E.2 - A.2) * (B.2 - E.2) = 0

-- Define the theorem
theorem right_triangle_circles_coincide 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_circle_BC : circle_BC B C D) 
  (h_circle_AB : circle_AB A B E) 
  (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 200)
  (h_AC : ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt = 40) :
  D = E := by
  sorry


end right_triangle_circles_coincide_l1744_174449


namespace unique_solution_for_diophantine_equation_l1744_174486

theorem unique_solution_for_diophantine_equation :
  ∀ m a b : ℤ,
    m > 1 ∧ a > 1 ∧ b > 1 →
    (m + 1) * a = m * b + 1 →
    m = 2 ∧ a = 3 ∧ b = 5 := by
  sorry

end unique_solution_for_diophantine_equation_l1744_174486


namespace right_triangle_area_l1744_174439

/-- A right triangle with vertices A(0, 0), B(0, 5), and C(3, 0) has an area of 7.5 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (3, 0)
  let triangle_area := (1/2) * 3 * 5
  triangle_area = 7.5 := by
  sorry

end right_triangle_area_l1744_174439


namespace smallest_positive_integer_ending_in_9_divisible_by_7_l1744_174476

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_7 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 7 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 7 = 0 → m ≥ n :=
by
  use 49
  sorry

end smallest_positive_integer_ending_in_9_divisible_by_7_l1744_174476


namespace factorization_problems_l1744_174413

theorem factorization_problems (a x y : ℝ) : 
  (a * (a - 2) + 2 * (a - 2) = (a - 2) * (a + 2)) ∧ 
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end factorization_problems_l1744_174413


namespace defective_probability_l1744_174469

theorem defective_probability (total_output : ℝ) : 
  let machine_a_output := 0.40 * total_output
  let machine_b_output := 0.35 * total_output
  let machine_c_output := 0.25 * total_output
  let machine_a_defective_rate := 14 / 2000
  let machine_b_defective_rate := 9 / 1500
  let machine_c_defective_rate := 7 / 1000
  let total_defective := 
    machine_a_defective_rate * machine_a_output +
    machine_b_defective_rate * machine_b_output +
    machine_c_defective_rate * machine_c_output
  total_defective / total_output = 0.00665 := by
sorry

end defective_probability_l1744_174469


namespace cost_to_fill_can_n_l1744_174495

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost in dollars to fill a given volume of gasoline -/
def fillCost (volume : ℝ) : ℝ := sorry

theorem cost_to_fill_can_n (can_b can_n : Cylinder) (half_b_cost : ℝ) : 
  can_n.radius = 2 * can_b.radius →
  can_n.height = can_b.height / 2 →
  fillCost (π * can_b.radius^2 * can_b.height / 2) = 4 →
  fillCost (π * can_n.radius^2 * can_n.height) = 16 := by sorry

end cost_to_fill_can_n_l1744_174495


namespace simultaneous_sound_arrival_l1744_174497

/-- Given a shooting range of length d meters, a bullet speed of c m/sec, and a speed of sound of s m/sec,
    the point x where the sound of the gunshot and the sound of the bullet hitting the target 
    arrive simultaneously is (d/2) * (1 + s/c) meters from the shooting position. -/
theorem simultaneous_sound_arrival (d c s : ℝ) (hd : d > 0) (hc : c > 0) (hs : s > 0) :
  let x := (d / 2) * (1 + s / c)
  (x / s) = (d / c + (d - x) / s) :=
by sorry

end simultaneous_sound_arrival_l1744_174497


namespace complex_fraction_simplification_l1744_174436

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5/17 - 14/17 * i := by
  sorry

end complex_fraction_simplification_l1744_174436


namespace negation_of_universal_proposition_l1744_174435

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 3 → x^3 - 27 > 0) ↔ (∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0) := by sorry

end negation_of_universal_proposition_l1744_174435


namespace class_composition_l1744_174414

theorem class_composition (initial_girls : ℕ) (initial_boys : ℕ) (girls_left : ℕ) :
  initial_girls * 6 = initial_boys * 5 →
  (initial_girls - girls_left) * 3 = initial_boys * 2 →
  girls_left = 20 →
  initial_boys = 120 := by
  sorry

end class_composition_l1744_174414


namespace triangle_area_is_five_l1744_174417

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 5

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle -/
def triangle_area : ℝ := 5

/-- Theorem: The area of the triangle formed by the line 2x - 5y - 10 = 0 and the coordinate axes is 5 -/
theorem triangle_area_is_five : 
  triangle_area = (1/2) * x_intercept * (-y_intercept) :=
by sorry

end triangle_area_is_five_l1744_174417


namespace dividend_calculation_l1744_174422

theorem dividend_calculation (divisor remainder quotient : ℕ) : 
  divisor = 17 → remainder = 8 → quotient = 4 → 
  divisor * quotient + remainder = 76 := by
sorry

end dividend_calculation_l1744_174422


namespace problem_solution_l1744_174496

theorem problem_solution (x y z : ℝ) (h1 : x + y + z = 25) (h2 : y + z = 14) : x = 11 := by
  sorry

end problem_solution_l1744_174496


namespace work_completion_time_l1744_174406

theorem work_completion_time 
  (john_time : ℝ) 
  (rose_time : ℝ) 
  (dave_time : ℝ) 
  (h1 : john_time = 8) 
  (h2 : rose_time = 16) 
  (h3 : dave_time = 12) : 
  (1 / (1 / john_time + 1 / rose_time + 1 / dave_time)) = 48 / 13 := by
  sorry

#check work_completion_time

end work_completion_time_l1744_174406


namespace meet_once_l1744_174488

/-- Represents the movement of Hannah and the van --/
structure Movement where
  hannah_speed : ℝ
  van_speed : ℝ
  pail_distance : ℝ
  van_stop_time : ℝ

/-- Calculates the number of meetings between Hannah and the van --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that Hannah and the van meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.hannah_speed = 6)
  (h2 : m.van_speed = 12)
  (h3 : m.pail_distance = 150)
  (h4 : m.van_stop_time = 45)
  : number_of_meetings m = 1 := by
  sorry

end meet_once_l1744_174488


namespace evaluate_expression_l1744_174438

theorem evaluate_expression : (16^24) / (32^12) = 8^12 := by
  sorry

end evaluate_expression_l1744_174438


namespace singing_competition_winner_l1744_174402

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students making guesses -/
inductive Student : Type
  | A | B | C | D

def guess (s : Student) (c : Contestant) : Prop :=
  match s with
  | Student.A => c = Contestant.four ∨ c = Contestant.five
  | Student.B => c ≠ Contestant.three
  | Student.C => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

theorem singing_competition_winner :
  ∃! (winner : Contestant),
    (∃! (correct_guesser : Student), guess correct_guesser winner) ∧
    (∀ (c : Contestant), c ≠ winner → ¬ guess Student.A c ∧ ¬ guess Student.B c ∧ ¬ guess Student.C c ∧ ¬ guess Student.D c) ∧
    winner = Contestant.three :=
by sorry

end singing_competition_winner_l1744_174402


namespace tetrahedron_volume_l1744_174441

/-- Given a tetrahedron with two faces of areas S₁ and S₂, sharing a common edge of length a,
    and with a dihedral angle α between these faces, the volume V of the tetrahedron is
    (2 * S₁ * S₂ * sin α) / (3 * a) -/
theorem tetrahedron_volume
  (S₁ S₂ a : ℝ)
  (α : ℝ)
  (h₁ : S₁ > 0)
  (h₂ : S₂ > 0)
  (h₃ : a > 0)
  (h₄ : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (2 * S₁ * S₂ * Real.sin α) / (3 * a) ∧ V > 0 := by
  sorry

end tetrahedron_volume_l1744_174441


namespace triangle_perimeter_l1744_174489

theorem triangle_perimeter (a : ℕ) (h1 : Odd a) (h2 : 3 < a) (h3 : a < 9) :
  (3 + 6 + a = 14) ∨ (3 + 6 + a = 16) := by
  sorry

end triangle_perimeter_l1744_174489
