import Mathlib

namespace infinitely_many_lcm_greater_than_ck_l3371_337100

theorem infinitely_many_lcm_greater_than_ck
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_positive : ∀ n, a n > 0)
  (c : ℝ)
  (h_c_bounds : 0 < c ∧ c < 3/2) :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ Nat.lcm (a k) (a (k + 1)) > ↑k * c :=
sorry

end infinitely_many_lcm_greater_than_ck_l3371_337100


namespace condition_relationship_l3371_337177

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a - b > 0 → a^2 - b^2 > 0) ∧
  (∃ a b, a^2 - b^2 > 0 ∧ a - b ≤ 0) :=
by sorry

end condition_relationship_l3371_337177


namespace circle_and_tangent_line_l3371_337147

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem circle_and_tangent_line 
  (C : Circle) 
  (l : Line) :
  C.h = 2 ∧ 
  C.k = 3 ∧ 
  C.r = 1 ∧
  l.x₀ = 1 ∧ 
  l.y₀ = 0 →
  (∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2) ∧
  ((l.a = 1 ∧ l.b = 0 ∧ l.c = -1) ∨
   (l.a = 4 ∧ l.b = -3 ∧ l.c = -4)) :=
by sorry

end circle_and_tangent_line_l3371_337147


namespace routes_from_p_to_q_l3371_337134

/-- Represents a directed graph with vertices P, R, S, T, Q -/
structure Network where
  vertices : Finset Char
  edges : Finset (Char × Char)

/-- Counts the number of paths between two vertices in the network -/
def count_paths (n : Network) (start finish : Char) : ℕ :=
  sorry

/-- The specific network described in the problem -/
def problem_network : Network :=
  { vertices := {'P', 'R', 'S', 'T', 'Q'},
    edges := {('P', 'R'), ('P', 'S'), ('P', 'T'), ('R', 'T'), ('R', 'Q'), ('S', 'R'), ('S', 'T'), ('S', 'Q'), ('T', 'R'), ('T', 'S'), ('T', 'Q')} }

theorem routes_from_p_to_q (n : Network := problem_network) :
  count_paths n 'P' 'Q' = 16 :=
sorry

end routes_from_p_to_q_l3371_337134


namespace quadratic_roots_negative_reciprocals_l3371_337104

theorem quadratic_roots_negative_reciprocals (k : ℝ) : 
  (∃ α : ℝ, α ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = α ∨ x = -1/α))) →
  k = -1 :=
by sorry

end quadratic_roots_negative_reciprocals_l3371_337104


namespace calculate_annual_interest_rate_l3371_337180

/-- Given an initial charge and the amount owed after one year with simple annual interest,
    calculate the annual interest rate. -/
theorem calculate_annual_interest_rate
  (initial_charge : ℝ)
  (amount_owed_after_year : ℝ)
  (h1 : initial_charge = 35)
  (h2 : amount_owed_after_year = 37.1)
  (h3 : amount_owed_after_year = initial_charge * (1 + interest_rate))
  : interest_rate = 0.06 :=
sorry

end calculate_annual_interest_rate_l3371_337180


namespace at_most_one_lattice_point_on_circle_l3371_337199

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem at_most_one_lattice_point_on_circle 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) 
  (p q : LatticePoint) 
  (r : ℝ) 
  (h_p : squaredDistance (p.x, p.y) center = r^2) 
  (h_q : squaredDistance (q.x, q.y) center = r^2) : 
  p = q :=
sorry

end at_most_one_lattice_point_on_circle_l3371_337199


namespace fly_distance_from_ceiling_l3371_337126

theorem fly_distance_from_ceiling (x y z : ℝ) : 
  x = 3 → y = 4 → (x^2 + y^2 + z^2 = 5^2) → z = 0 := by sorry

end fly_distance_from_ceiling_l3371_337126


namespace find_divisor_l3371_337191

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 23) (h2 : quotient = 4) (h3 : remainder = 3) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 5 := by
sorry

end find_divisor_l3371_337191


namespace fifty_percent_greater_than_88_l3371_337152

theorem fifty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.5 → x = 132 := by
  sorry

end fifty_percent_greater_than_88_l3371_337152


namespace total_monthly_earnings_l3371_337133

/-- Represents an apartment floor --/
structure Floor :=
  (rooms : ℕ)
  (rent : ℝ)
  (occupancy : ℝ)

/-- Calculates the monthly earnings for a floor --/
def floorEarnings (f : Floor) : ℝ :=
  f.rooms * f.rent * f.occupancy

/-- Represents an apartment building --/
structure Building :=
  (floors : List Floor)

/-- Calculates the total monthly earnings for a building --/
def buildingEarnings (b : Building) : ℝ :=
  (b.floors.map floorEarnings).sum

/-- The first building --/
def building1 : Building :=
  { floors := [
    { rooms := 5, rent := 15, occupancy := 0.8 },
    { rooms := 6, rent := 25, occupancy := 0.75 },
    { rooms := 9, rent := 30, occupancy := 0.5 },
    { rooms := 4, rent := 60, occupancy := 0.85 }
  ] }

/-- The second building --/
def building2 : Building :=
  { floors := [
    { rooms := 7, rent := 20, occupancy := 0.9 },
    { rooms := 8, rent := 42.5, occupancy := 0.7 }, -- Average rent for the second floor
    { rooms := 6, rent := 60, occupancy := 0.6 }
  ] }

/-- The main theorem --/
theorem total_monthly_earnings :
  buildingEarnings building1 + buildingEarnings building2 = 1091.5 := by
  sorry

end total_monthly_earnings_l3371_337133


namespace leonards_age_l3371_337169

theorem leonards_age (leonard nina jerome : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : leonard + nina + jerome = 36) :
  leonard = 6 := by
sorry

end leonards_age_l3371_337169


namespace problem_statement_l3371_337159

def f (x : ℝ) := x^3 - x^2

theorem problem_statement :
  (∀ m n : ℝ, m > 0 → n > 0 → m * n > 1 → max (f m) (f n) ≥ 0) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → f a = f b → a + b > 1) :=
by sorry

end problem_statement_l3371_337159


namespace smallest_three_digit_congruence_l3371_337163

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    75 * n % 345 = 225 ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 75 * m % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end smallest_three_digit_congruence_l3371_337163


namespace percentage_calculation_l3371_337115

theorem percentage_calculation (x : ℝ) (h : 0.25 * x = 1200) : 0.35 * x = 1680 := by
  sorry

end percentage_calculation_l3371_337115


namespace christine_savings_theorem_l3371_337175

/-- Calculates Christine's savings for the month based on her sales and commission structure -/
def christine_savings (
  electronics_rate : ℚ)
  (clothing_rate : ℚ)
  (furniture_rate : ℚ)
  (domestic_electronics : ℚ)
  (domestic_clothing : ℚ)
  (domestic_furniture : ℚ)
  (international_electronics : ℚ)
  (international_clothing : ℚ)
  (international_furniture : ℚ)
  (exchange_rate : ℚ)
  (tax_rate : ℚ)
  (personal_needs_rate : ℚ)
  (investment_rate : ℚ) : ℚ :=
  let domestic_commission := 
    electronics_rate * domestic_electronics +
    clothing_rate * domestic_clothing +
    furniture_rate * domestic_furniture
  let international_commission := 
    (electronics_rate * international_electronics +
    clothing_rate * international_clothing +
    furniture_rate * international_furniture) * exchange_rate
  let tax := international_commission * tax_rate
  let post_tax_international := international_commission - tax
  let international_savings := 
    post_tax_international * (1 - personal_needs_rate - investment_rate)
  domestic_commission + international_savings

theorem christine_savings_theorem :
  christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30 = 3579.4375 := by
  sorry

#eval christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30

end christine_savings_theorem_l3371_337175


namespace remaining_red_balloons_l3371_337119

/-- The number of red balloons remaining after destruction --/
def remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - destroyed_balloons

/-- Theorem stating the number of remaining red balloons --/
theorem remaining_red_balloons :
  remaining_balloons 10.0 46.0 16.0 = 40.0 := by
  sorry

end remaining_red_balloons_l3371_337119


namespace expand_and_simplify_l3371_337146

theorem expand_and_simplify (x : ℝ) : 2*(x+3)*(x^2 + 2*x + 7) = 2*x^3 + 10*x^2 + 26*x + 42 := by
  sorry

end expand_and_simplify_l3371_337146


namespace carol_extra_invitations_l3371_337162

def invitation_problem (packs_bought : ℕ) (invitations_per_pack : ℕ) (friends_to_invite : ℕ) : ℕ :=
  let total_invitations := packs_bought * invitations_per_pack
  let additional_packs_needed := ((friends_to_invite - total_invitations) + invitations_per_pack - 1) / invitations_per_pack
  let final_invitations := total_invitations + additional_packs_needed * invitations_per_pack
  final_invitations - friends_to_invite

theorem carol_extra_invitations :
  invitation_problem 3 5 23 = 2 :=
sorry

end carol_extra_invitations_l3371_337162


namespace blakes_change_is_correct_l3371_337154

/-- Calculates the change Blake receives after buying candy with discounts -/
def blakes_change (lollipop_price : ℚ) (gummy_price : ℚ) (candy_bar_price : ℚ) : ℚ :=
  let chocolate_price := 4 * lollipop_price
  let lollipop_cost := 3 * lollipop_price + lollipop_price / 2
  let chocolate_cost := 4 * chocolate_price + 2 * (chocolate_price * 3 / 4)
  let gummy_cost := 3 * gummy_price
  let candy_bar_cost := 5 * candy_bar_price
  let total_cost := lollipop_cost + chocolate_cost + gummy_cost + candy_bar_cost
  let total_given := 4 * 20 + 2 * 5 + 5 * 1
  total_given - total_cost

/-- Theorem stating that Blake's change is $27.50 -/
theorem blakes_change_is_correct :
  blakes_change 2 3 (3/2) = 55/2 := by sorry

end blakes_change_is_correct_l3371_337154


namespace arithmetic_sequence_sum_l3371_337113

theorem arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 1 →
  d = 2 →
  n > 0 →
  aₙ = 21 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 121 := by
  sorry

end arithmetic_sequence_sum_l3371_337113


namespace expression_simplification_l3371_337161

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  let f := (((x^2 - x) / (x^2 - 2*x + 1) + 2 / (x - 1)) / ((x^2 - 4) / (x^2 - 1)))
  x = 3 → f = 4 := by sorry

end expression_simplification_l3371_337161


namespace exhibition_spacing_l3371_337118

theorem exhibition_spacing (wall_width : ℕ) (painting_width : ℕ) (num_paintings : ℕ) :
  wall_width = 320 ∧ painting_width = 30 ∧ num_paintings = 6 →
  (wall_width - num_paintings * painting_width) / (num_paintings + 1) = 20 :=
by sorry

end exhibition_spacing_l3371_337118


namespace remainder_problem_l3371_337182

theorem remainder_problem (N : ℤ) : 
  N % 37 = 1 → N % 296 = 260 := by
  sorry

end remainder_problem_l3371_337182


namespace union_subset_iff_m_in_range_l3371_337192

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x : ℝ | m * x + 1 > 0}

theorem union_subset_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B) ⊆ C m ↔ m ∈ Set.Icc (-1/2) 1 := by sorry

end union_subset_iff_m_in_range_l3371_337192


namespace line_xz_plane_intersection_l3371_337183

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_xz_plane_intersection (p₁ p₂ q : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 5) → 
  p₂ = (4, 0, 9) → 
  (∃ t : ℝ, q = p₁ + t • (p₂ - p₁)) → 
  q.2 = 0 → 
  q = (4, 0, 9) := by
  sorry

#check line_xz_plane_intersection

end line_xz_plane_intersection_l3371_337183


namespace fraction_simplification_l3371_337164

theorem fraction_simplification (x y : ℝ) : 
  (2*x + y)/4 + (5*y - 4*x)/6 - y/12 = (-x + 6*y)/6 := by sorry

end fraction_simplification_l3371_337164


namespace keith_digimon_pack_price_l3371_337141

/-- The price of each pack of Digimon cards -/
def digimon_pack_price (total_spent : ℚ) (baseball_deck_price : ℚ) (num_digimon_packs : ℕ) : ℚ :=
  (total_spent - baseball_deck_price) / num_digimon_packs

theorem keith_digimon_pack_price :
  digimon_pack_price 23.86 6.06 4 = 4.45 := by
  sorry

end keith_digimon_pack_price_l3371_337141


namespace marks_songs_per_gig_l3371_337179

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents the number of gigs Mark does in two weeks -/
def number_of_gigs : ℕ := days_in_two_weeks / 2

/-- Represents the duration of a short song in minutes -/
def short_song_duration : ℕ := 5

/-- Represents the duration of a long song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- Represents the number of short songs per gig -/
def short_songs_per_gig : ℕ := 2

/-- Represents the number of long songs per gig -/
def long_songs_per_gig : ℕ := 1

/-- Represents the total playing time for all gigs in minutes -/
def total_playing_time : ℕ := 280

/-- Theorem: Given the conditions, Mark plays 7 songs at each gig -/
theorem marks_songs_per_gig :
  ∃ (songs_per_gig : ℕ),
    songs_per_gig = short_songs_per_gig + long_songs_per_gig +
      ((total_playing_time / number_of_gigs) -
       (short_songs_per_gig * short_song_duration + long_songs_per_gig * long_song_duration)) /
      short_song_duration ∧
    songs_per_gig = 7 :=
by sorry

end marks_songs_per_gig_l3371_337179


namespace consecutive_integers_sqrt_three_sum_l3371_337160

theorem consecutive_integers_sqrt_three_sum (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end consecutive_integers_sqrt_three_sum_l3371_337160


namespace area_of_two_sectors_l3371_337178

/-- The area of a figure formed by two sectors of a circle -/
theorem area_of_two_sectors (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 10) (h2 : angle1 = 45) (h3 : angle2 = 90) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 37.5 * π := by
  sorry

end area_of_two_sectors_l3371_337178


namespace divisibility_by_1956_l3371_337122

theorem divisibility_by_1956 (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, 24 * 80^n + 1992 * 83^(n-1) = 1956 * k := by
  sorry

end divisibility_by_1956_l3371_337122


namespace platform_length_calculation_l3371_337176

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 54 →
  time_pole = 18 →
  ∃ platform_length : ℝ, abs (platform_length - 600.18) < 0.01 := by
  sorry

#check platform_length_calculation

end platform_length_calculation_l3371_337176


namespace negation_equivalence_l3371_337195

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end negation_equivalence_l3371_337195


namespace daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l3371_337121

-- Define the variables and constants
variable (x : ℝ) -- Selling price in yuan
variable (y : ℝ) -- Daily sales volume in items
variable (w : ℝ) -- Daily sales profit in yuan

-- Define the given conditions
def cost_price : ℝ := 6
def min_price : ℝ := 6
def max_price : ℝ := 12
def base_price : ℝ := 8
def base_volume : ℝ := 200
def volume_change_rate : ℝ := 10

-- Theorem 1: Daily sales volume function
theorem daily_sales_volume : 
  ∀ x, min_price ≤ x ∧ x ≤ max_price → y = -volume_change_rate * x + (base_volume + volume_change_rate * base_price) :=
sorry

-- Theorem 2: Selling price for specific profit
theorem selling_price_for_profit (target_profit : ℝ) : 
  ∃ x, min_price ≤ x ∧ x ≤ max_price ∧ 
  (x - cost_price) * (-volume_change_rate * x + (base_volume + volume_change_rate * base_price)) = target_profit :=
sorry

-- Theorem 3: Daily sales profit function and maximum profit
theorem daily_sales_profit_and_max : 
  ∃ w_max : ℝ,
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    w = -volume_change_rate * (x - 11)^2 + 1210) ∧
  (w_max = -volume_change_rate * (max_price - 11)^2 + 1210) ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → w ≤ w_max) :=
sorry

end daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l3371_337121


namespace fayes_coloring_books_l3371_337132

theorem fayes_coloring_books : 
  ∀ (initial_books : ℕ), 
  (initial_books - 3 + 48 = 79) → initial_books = 34 :=
by
  sorry

end fayes_coloring_books_l3371_337132


namespace hyperbola_asymptote_angle_l3371_337114

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ θ : ℝ, θ = Real.pi/4 ∧ 
    (∀ t : ℝ, ∃ x y : ℝ, x = t ∧ y = (b/a)*t ∨ x = t ∧ y = -(b/a)*t) ∧
    θ = Real.arctan ((2*(b/a))/(1 - (b/a)^2))) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end hyperbola_asymptote_angle_l3371_337114


namespace logistics_problem_l3371_337110

/-- Represents the freight rates and charges for a logistics company. -/
structure FreightData where
  rateA : ℝ  -- Freight rate for goods A
  rateB : ℝ  -- Freight rate for goods B
  totalCharge : ℝ  -- Total freight charge

/-- Calculates the quantities of goods A and B transported given freight data for two months. -/
def calculateQuantities (march : FreightData) (april : FreightData) : ℝ × ℝ :=
  sorry

/-- Theorem stating that given the specific freight data for March and April,
    the quantities of goods A and B transported are 100 tons and 140 tons respectively. -/
theorem logistics_problem (march : FreightData) (april : FreightData) 
  (h1 : march.rateA = 50)
  (h2 : march.rateB = 30)
  (h3 : march.totalCharge = 9500)
  (h4 : april.rateA = 70)  -- 50 * 1.4 = 70
  (h5 : april.rateB = 40)
  (h6 : april.totalCharge = 13000) :
  calculateQuantities march april = (100, 140) :=
sorry

end logistics_problem_l3371_337110


namespace orange_juice_division_l3371_337108

theorem orange_juice_division (total_pints : ℚ) (num_glasses : ℕ) 
  (h1 : total_pints = 153)
  (h2 : num_glasses = 5) :
  total_pints / num_glasses = 30.6 := by
  sorry

end orange_juice_division_l3371_337108


namespace eighth_term_of_sequence_l3371_337139

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmeticSequence a₁ d 4 = 22 →
  arithmeticSequence a₁ d 6 = 46 →
  arithmeticSequence a₁ d 8 = 70 := by
sorry

end eighth_term_of_sequence_l3371_337139


namespace friendly_point_sum_l3371_337135

/-- Friendly point transformation in 2D plane -/
def friendly_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 - 1, -p.1 - 1)

/-- Sequence of friendly points -/
def friendly_sequence (start : ℝ × ℝ) : ℕ → ℝ × ℝ
| 0 => start
| n + 1 => friendly_point (friendly_sequence start n)

theorem friendly_point_sum (x y : ℝ) :
  friendly_sequence (x, y) 2022 = (-3, -2) →
  x + y = 3 :=
by sorry

end friendly_point_sum_l3371_337135


namespace strawberry_pies_l3371_337129

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_picked : ℕ) (rachel_factor : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_picked + christine_picked * rachel_factor) / pounds_per_pie

/-- Theorem stating that Christine and Rachel can make 10 pies -/
theorem strawberry_pies :
  number_of_pies 10 2 3 = 10 := by
  sorry

end strawberry_pies_l3371_337129


namespace zara_brixton_height_l3371_337142

/-- The heights of four people satisfying certain conditions -/
structure Heights where
  itzayana : ℝ
  zora : ℝ
  brixton : ℝ
  zara : ℝ
  itzayana_taller : itzayana = zora + 4
  zora_shorter : zora = brixton - 8
  zara_equal : zara = brixton
  average_height : (itzayana + zora + brixton + zara) / 4 = 61

/-- Theorem stating that Zara and Brixton's height is 64 inches -/
theorem zara_brixton_height (h : Heights) : h.zara = 64 ∧ h.brixton = 64 := by
  sorry

end zara_brixton_height_l3371_337142


namespace imaginary_part_of_z_l3371_337193

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = 2) : 
  z.im = -1 := by
  sorry

end imaginary_part_of_z_l3371_337193


namespace smallest_gcd_bc_l3371_337166

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  (∀ b' c' : ℕ+, Nat.gcd a b' = 294 → Nat.gcd a c' = 1155 → Nat.gcd b c ≤ Nat.gcd b' c') ∧
  Nat.gcd b c = 21 :=
sorry

end smallest_gcd_bc_l3371_337166


namespace mixture_alcohol_percentage_l3371_337140

/-- The percentage of alcohol in solution X -/
def alcohol_percent_X : ℝ := 15

/-- The percentage of alcohol in solution Y -/
def alcohol_percent_Y : ℝ := 45

/-- The initial volume of solution X in milliliters -/
def initial_volume_X : ℝ := 300

/-- The volume of solution Y to be added in milliliters -/
def volume_Y : ℝ := 150

/-- The desired percentage of alcohol in the final solution -/
def target_alcohol_percent : ℝ := 25

/-- Theorem stating that adding 150 mL of solution Y to 300 mL of solution X
    results in a solution with 25% alcohol by volume -/
theorem mixture_alcohol_percentage :
  let total_volume := initial_volume_X + volume_Y
  let total_alcohol := (alcohol_percent_X / 100) * initial_volume_X + (alcohol_percent_Y / 100) * volume_Y
  (total_alcohol / total_volume) * 100 = target_alcohol_percent := by
  sorry

end mixture_alcohol_percentage_l3371_337140


namespace area_of_quadrilateral_ABCD_l3371_337171

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents the quadrilateral ABCD formed by the intersection of a plane with the cube -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the area of the quadrilateral ABCD -/
def quadrilateralArea (quad : Quadrilateral) : ℝ := sorry

/-- Main theorem: The area of quadrilateral ABCD is 2√3 -/
theorem area_of_quadrilateral_ABCD :
  let cube := Cube.mk 2
  let A := Point3D.mk 0 0 0
  let C := Point3D.mk 2 2 2
  let B := Point3D.mk (2/3) 2 0
  let D := Point3D.mk 2 (4/3) 2
  let quad := Quadrilateral.mk A B C D
  quadrilateralArea quad = 2 * Real.sqrt 3 := by sorry

end area_of_quadrilateral_ABCD_l3371_337171


namespace distinct_colorings_l3371_337117

/-- The number of disks in the circle -/
def n : ℕ := 7

/-- The number of blue disks -/
def blue : ℕ := 3

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The total number of colorings without considering symmetries -/
def total_colorings : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The theorem stating the number of distinct colorings -/
theorem distinct_colorings : 
  (total_colorings / symmetries : ℚ) = 20 := by sorry

end distinct_colorings_l3371_337117


namespace triangle_theorem_l3371_337105

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  t.C = π / 3 ∧ 
  (t.a = 5 → t.b = 8 → t.c = 7) := by
  sorry

-- Note: The proof is omitted as per the instructions

end triangle_theorem_l3371_337105


namespace simplify_expression_l3371_337197

theorem simplify_expression (a : ℝ) : (36 * a ^ 9) ^ 4 * (63 * a ^ 9) ^ 4 = a ^ 4 := by
  sorry

end simplify_expression_l3371_337197


namespace investment_period_l3371_337109

/-- Proves that given a sum of 7000 invested at 15% p.a. and 12% p.a., 
    if the difference in interest received is 420, then the investment period is 2 years. -/
theorem investment_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 7000 →
  rate_high = 0.15 →
  rate_low = 0.12 →
  interest_diff = 420 →
  ∃ (years : ℝ), principal * rate_high * years - principal * rate_low * years = interest_diff ∧ years = 2 :=
by sorry

end investment_period_l3371_337109


namespace largest_unrepresentable_amount_is_correct_l3371_337149

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {6*n + 1, 6*n + 4, 6*n + 7, 6*n + 10}

/-- Predicate to check if an amount can be represented using given coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), s = a*(6*n + 1) + b*(6*n + 4) + c*(6*n + 7) + d*(6*n + 10)

/-- The largest amount that cannot be represented using the given coin denominations -/
def largest_unrepresentable_amount (n : ℕ) : ℕ :=
  12*n^2 + 14*n - 1

/-- Theorem stating that the largest_unrepresentable_amount is correct -/
theorem largest_unrepresentable_amount_is_correct (n : ℕ) :
  (∀ k < largest_unrepresentable_amount n, is_representable k n) ∧
  ¬is_representable (largest_unrepresentable_amount n) n :=
by sorry

end largest_unrepresentable_amount_is_correct_l3371_337149


namespace smallest_divisor_sum_of_squares_l3371_337188

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ (a b : ℕ), a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end smallest_divisor_sum_of_squares_l3371_337188


namespace geometric_sequence_existence_l3371_337155

theorem geometric_sequence_existence : ∃ (a r : ℝ), 
  a * r = 2 ∧ 
  a * r^3 = 6 ∧ 
  a = -2 * Real.sqrt 3 / 3 := by
sorry

end geometric_sequence_existence_l3371_337155


namespace value_of_A_l3371_337101

-- Define the letter values as variables
variable (F L A G E : ℤ)

-- Define the given conditions
axiom G_value : G = 15
axiom FLAG_value : F + L + A + G = 50
axiom LEAF_value : L + E + A + F = 65
axiom FEEL_value : F + E + E + L = 58

-- Theorem to prove
theorem value_of_A : A = 37 := by
  sorry

end value_of_A_l3371_337101


namespace revolver_game_theorem_l3371_337123

/-- The probability that player A fires the bullet in the revolver game -/
def revolver_game_prob : ℚ :=
  let p : ℚ := 1/6  -- probability of firing on a single shot
  6/11

/-- The revolver game theorem -/
theorem revolver_game_theorem :
  let p : ℚ := 1/6  -- probability of firing on a single shot
  let q : ℚ := 1 - p  -- probability of not firing on a single shot
  revolver_game_prob = p / (1 - q^2) :=
by sorry

#eval revolver_game_prob

end revolver_game_theorem_l3371_337123


namespace atheris_population_2080_l3371_337131

def population_growth (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * (4 ^ (years / 30))

theorem atheris_population_2080 :
  population_growth 250 80 = 4000 := by
  sorry

end atheris_population_2080_l3371_337131


namespace september_reading_goal_l3371_337116

def total_pages_read (total_days : ℕ) (non_reading_days : ℕ) (special_day_pages : ℕ) (regular_daily_pages : ℕ) : ℕ :=
  let reading_days := total_days - non_reading_days
  let regular_reading_days := reading_days - 1
  regular_reading_days * regular_daily_pages + special_day_pages

theorem september_reading_goal :
  total_pages_read 30 4 100 20 = 600 := by
  sorry

end september_reading_goal_l3371_337116


namespace expression_evaluation_l3371_337144

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l3371_337144


namespace value_of_c_l3371_337165

theorem value_of_c (a b c : ℝ) 
  (h1 : 12 = 0.06 * a) 
  (h2 : 6 = 0.12 * b) 
  (h3 : c = b / a) : 
  c = 0.25 := by
  sorry

end value_of_c_l3371_337165


namespace hyperbola_m_range_l3371_337153

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (1 + m) + y^2 / (1 - m) = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m ∧ 
  (1 + m > 0 ∧ 1 - m < 0) ∨ (1 + m < 0 ∧ 1 - m > 0)

-- Theorem stating the range of m for which the equation represents a hyperbola
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m < -1 ∨ m > 1 :=
sorry

end hyperbola_m_range_l3371_337153


namespace oddProbabilityConvergesTo1Third_l3371_337170

/-- Represents the state of the calculator --/
structure CalculatorState where
  display : ℕ
  lastOperation : Option (ℕ → ℕ → ℕ)

/-- Represents a button press on the calculator --/
inductive ButtonPress
  | Digit (d : Fin 10)
  | Add
  | Multiply

/-- The probability of the display showing an odd number after n button presses --/
def oddProbability (n : ℕ) : ℝ := sorry

/-- The limiting probability of the display showing an odd number as n approaches infinity --/
def limitingOddProbability : ℝ := sorry

/-- The main theorem stating that the limiting probability converges to 1/3 --/
theorem oddProbabilityConvergesTo1Third :
  limitingOddProbability = 1/3 := by sorry

end oddProbabilityConvergesTo1Third_l3371_337170


namespace largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l3371_337138

theorem largest_n_divisible_by_seven (n : ℕ) : 
  (n < 200000 ∧ 
   (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0) →
  n ≤ 199999 :=
by sorry

theorem n_199999_satisfies_condition : 
  (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 :=
by sorry

theorem n_199999_is_largest : 
  ∀ m : ℕ, m < 200000 ∧ 
  (8 * (m - 3)^5 - 2 * m^2 + 18 * m - 36) % 7 = 0 →
  m ≤ 199999 :=
by sorry

end largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l3371_337138


namespace leap_year_statistics_l3371_337128

def leap_year_data : List ℕ := sorry

def median_of_modes (data : List ℕ) : ℚ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem leap_year_statistics :
  let d := median_of_modes leap_year_data
  let M := median leap_year_data
  let μ := mean leap_year_data
  d < M ∧ M < μ := by sorry

end leap_year_statistics_l3371_337128


namespace root_product_identity_l3371_337106

theorem root_product_identity (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end root_product_identity_l3371_337106


namespace unique_point_equal_angles_l3371_337189

/-- The ellipse equation x²/4 + y² = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The focus F = (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- A chord AB passing through F -/
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = (2 + t * (B.1 - 2), t * B.2) ∧ 
             is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

/-- Angles APF and BPF are equal -/
def equal_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1))^2 = (B.2 / (B.1 - P.1))^2

/-- The main theorem -/
theorem unique_point_equal_angles :
  ∃! (p : ℝ), p > 0 ∧ 
    (∀ (A B : ℝ × ℝ), is_chord_through_F A B → 
      equal_angles (p, 0) A B) ∧ 
    p = 2 := by sorry

end unique_point_equal_angles_l3371_337189


namespace f_behavior_at_infinity_l3371_337111

def f (x : ℝ) := -3 * x^4 + 4 * x^2 + 5

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x < M) :=
sorry

end f_behavior_at_infinity_l3371_337111


namespace intersection_values_l3371_337127

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the condition for intersection at a single point
def single_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ p.2 = line k p.1

-- Theorem statement
theorem intersection_values :
  {k : ℝ | single_intersection k} = {-1, 1, -Real.sqrt 5 / 2, Real.sqrt 5 / 2} :=
sorry

end intersection_values_l3371_337127


namespace days_worked_by_c_l3371_337158

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1480

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ 
  | 0 => 3
  | 1 => 4
  | 2 => 5

/-- 
Proves that given the conditions, the number of days worked by person c is 4
-/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end days_worked_by_c_l3371_337158


namespace quadratic_root_range_l3371_337107

theorem quadratic_root_range (a b : ℝ) (h1 : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ x^2 + a*x + 2*b - 2 = 0 ∧ y^2 + a*y + 2*b - 2 = 0) :
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 := by
sorry

end quadratic_root_range_l3371_337107


namespace boat_distance_downstream_l3371_337181

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 4) : 
  boat_speed + stream_speed * time = 112 := by
  sorry


end boat_distance_downstream_l3371_337181


namespace simple_interest_principal_l3371_337145

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * (rate / 100) * time →
  rate = 6.666666666666667 →
  time = 4 →
  interest = 160 →
  principal = 600 := by
sorry

end simple_interest_principal_l3371_337145


namespace choose_15_3_l3371_337184

theorem choose_15_3 : Nat.choose 15 3 = 455 := by sorry

end choose_15_3_l3371_337184


namespace no_intersection_l3371_337120

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- Define what it means for two functions to intersect at a point
def intersect_at (f g : ℝ → ℝ) (x : ℝ) : Prop := f x = g x

-- Theorem statement
theorem no_intersection :
  ¬ ∃ x : ℝ, intersect_at f g x :=
sorry

end no_intersection_l3371_337120


namespace yellow_bead_cost_l3371_337198

/-- The cost of a box of yellow beads, given the following conditions:
  * Red beads cost $1.30 per box
  * 10 boxes of mixed beads cost $1.72 per box
  * 4 boxes of each color (red and yellow) are used to make the 10 mixed boxes
-/
theorem yellow_bead_cost (red_cost : ℝ) (mixed_cost : ℝ) (red_boxes : ℕ) (yellow_boxes : ℕ) :
  red_cost = 1.30 →
  mixed_cost = 1.72 →
  red_boxes = 4 →
  yellow_boxes = 4 →
  red_boxes * red_cost + yellow_boxes * (3 : ℝ) = 10 * mixed_cost :=
by sorry

end yellow_bead_cost_l3371_337198


namespace root_multiplicity_two_l3371_337143

variable (n : ℕ)

def f (A B : ℝ) (x : ℝ) : ℝ := A * x^(n+1) + B * x^n + 1

theorem root_multiplicity_two (A B : ℝ) :
  (f n A B 1 = 0 ∧ (deriv (f n A B)) 1 = 0) ↔ (A = n ∧ B = -(n+1)) := by sorry

end root_multiplicity_two_l3371_337143


namespace coffee_consumption_ratio_l3371_337185

/-- Represents the number of coffees John used to buy daily -/
def old_coffee_count : ℕ := 4

/-- Represents the original price of each coffee in dollars -/
def old_coffee_price : ℚ := 2

/-- Represents the percentage increase in coffee price -/
def price_increase_percent : ℚ := 50

/-- Represents the amount John saves daily compared to his old spending in dollars -/
def daily_savings : ℚ := 2

/-- Theorem stating that the ratio of John's current coffee consumption to his previous consumption is 1:2 -/
theorem coffee_consumption_ratio :
  ∃ (new_coffee_count : ℕ),
    new_coffee_count * (old_coffee_price * (1 + price_increase_percent / 100)) = 
      old_coffee_count * old_coffee_price - daily_savings ∧
    new_coffee_count * 2 = old_coffee_count := by
  sorry

end coffee_consumption_ratio_l3371_337185


namespace roller_coaster_capacity_l3371_337148

theorem roller_coaster_capacity 
  (total_cars : ℕ) 
  (total_capacity : ℕ) 
  (four_seater_cars : ℕ) 
  (four_seater_capacity : ℕ) 
  (h1 : total_cars = 15)
  (h2 : total_capacity = 72)
  (h3 : four_seater_cars = 9)
  (h4 : four_seater_capacity = 4) :
  (total_capacity - four_seater_cars * four_seater_capacity) / (total_cars - four_seater_cars) = 6 := by
sorry

end roller_coaster_capacity_l3371_337148


namespace average_students_is_fifty_l3371_337112

/-- Represents a teacher's teaching data over multiple years -/
structure TeacherData where
  total_years : Nat
  first_year_students : Nat
  total_students : Nat

/-- Calculates the average number of students taught per year, excluding the first year -/
def averageStudentsPerYear (data : TeacherData) : Nat :=
  (data.total_students - data.first_year_students) / (data.total_years - 1)

/-- Theorem stating that for the given conditions, the average number of students per year (excluding the first year) is 50 -/
theorem average_students_is_fifty :
  let data : TeacherData := {
    total_years := 10,
    first_year_students := 40,
    total_students := 490
  }
  averageStudentsPerYear data = 50 := by
  sorry

#eval averageStudentsPerYear {
  total_years := 10,
  first_year_students := 40,
  total_students := 490
}

end average_students_is_fifty_l3371_337112


namespace expected_girls_left_of_boys_l3371_337157

/-- The number of boys in the lineup -/
def num_boys : ℕ := 10

/-- The number of girls in the lineup -/
def num_girls : ℕ := 7

/-- The total number of students in the lineup -/
def total_students : ℕ := num_boys + num_girls

/-- The expected number of girls standing to the left of all boys -/
def expected_girls_left : ℚ := 7 / 11

theorem expected_girls_left_of_boys :
  let random_arrangement := (Finset.range total_students).powerset
  expected_girls_left = (num_girls : ℚ) / (total_students + 1 : ℚ) := by sorry

end expected_girls_left_of_boys_l3371_337157


namespace solution_set_inequality_l3371_337187

theorem solution_set_inequality (x : ℝ) : (x - 2) / x < 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end solution_set_inequality_l3371_337187


namespace parabola_vertex_y_coordinate_l3371_337124

/-- The y-coordinate of the vertex of the parabola y = 2x^2 + 16x + 35 is 3 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 35
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 3 :=
by sorry

end parabola_vertex_y_coordinate_l3371_337124


namespace parallel_intersections_l3371_337156

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection of a plane with another plane resulting in a line
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem parallel_intersections
  (P1 P2 P3 : Plane) (l1 l2 : Line)
  (h1 : parallel_planes P1 P2)
  (h2 : l1 = intersect P3 P1)
  (h3 : l2 = intersect P3 P2) :
  parallel_lines l1 l2 := by
  sorry

end parallel_intersections_l3371_337156


namespace wire_cutting_l3371_337130

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 80 →
  ratio = 3 / 5 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 50 := by
sorry

end wire_cutting_l3371_337130


namespace x_coordinate_of_first_point_l3371_337167

/-- Given two points on a line, prove the x-coordinate of the first point -/
theorem x_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 := by
sorry

end x_coordinate_of_first_point_l3371_337167


namespace area_of_triangle_def_is_nine_l3371_337174

/-- A triangle with vertices on the sides of a rectangle -/
structure TriangleInRectangle where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- x-coordinate of vertex D -/
  dx : ℝ
  /-- y-coordinate of vertex D -/
  dy : ℝ
  /-- x-coordinate of vertex E -/
  ex : ℝ
  /-- y-coordinate of vertex E -/
  ey : ℝ
  /-- x-coordinate of vertex F -/
  fx : ℝ
  /-- y-coordinate of vertex F -/
  fy : ℝ
  /-- Ensure D is on the left side of the rectangle -/
  hd : dx = 0 ∧ 0 ≤ dy ∧ dy ≤ height
  /-- Ensure E is on the bottom side of the rectangle -/
  he : ey = 0 ∧ 0 ≤ ex ∧ ex ≤ width
  /-- Ensure F is on the top side of the rectangle -/
  hf : fy = height ∧ 0 ≤ fx ∧ fx ≤ width

/-- Calculate the area of the triangle DEF -/
def areaOfTriangleDEF (t : TriangleInRectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle DEF is 9 square units -/
theorem area_of_triangle_def_is_nine (t : TriangleInRectangle) 
    (h_width : t.width = 6) 
    (h_height : t.height = 4)
    (h_d : t.dx = 0 ∧ t.dy = 2)
    (h_e : t.ex = 6 ∧ t.ey = 0)
    (h_f : t.fx = 3 ∧ t.fy = 4) : 
  areaOfTriangleDEF t = 9 := by
  sorry

end area_of_triangle_def_is_nine_l3371_337174


namespace probability_A_selected_l3371_337168

/-- The number of individuals in the group -/
def n : ℕ := 3

/-- The number of representatives to be chosen -/
def k : ℕ := 2

/-- The probability of selecting A as one of the representatives -/
def prob_A_selected : ℚ := 2/3

/-- Theorem stating that the probability of selecting A as one of two representatives
    from a group of three individuals is 2/3 -/
theorem probability_A_selected :
  prob_A_selected = 2/3 := by sorry

end probability_A_selected_l3371_337168


namespace honda_production_l3371_337137

/-- Honda car production problem -/
theorem honda_production (day_shift second_shift total : ℕ) : 
  day_shift = 4 * second_shift → 
  second_shift = 1100 → 
  total = day_shift + second_shift → 
  total = 5500 := by
  sorry

end honda_production_l3371_337137


namespace inner_diagonal_sum_bound_l3371_337190

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- Sum of the lengths of the diagonals -/
  diagonalSum : ℝ
  /-- Convexity condition -/
  convex : diagonalSum > 0

/-- Theorem: For any two convex quadrilaterals where one is inside the other,
    the sum of the diagonals of the inner quadrilateral is less than twice
    the sum of the diagonals of the outer quadrilateral -/
theorem inner_diagonal_sum_bound
  (outer inner : ConvexQuadrilateral)
  (h : inner.diagonalSum < outer.diagonalSum) :
  inner.diagonalSum < 2 * outer.diagonalSum :=
by
  sorry


end inner_diagonal_sum_bound_l3371_337190


namespace ceiling_floor_calculation_l3371_337151

theorem ceiling_floor_calculation : 
  ⌈(12 / 5 : ℚ) * (((-19 : ℚ) / 4) - 3)⌉ - ⌊(12 / 5 : ℚ) * ⌊(-19 : ℚ) / 4⌋⌋ = -6 :=
by sorry

end ceiling_floor_calculation_l3371_337151


namespace existence_of_increasing_pair_l3371_337173

theorem existence_of_increasing_pair {α : Type*} [LinearOrder α] (a b : ℕ → α) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q :=
sorry

end existence_of_increasing_pair_l3371_337173


namespace minimize_sample_variance_l3371_337136

/-- Given a sample of size 5 with specific conditions, prove that the sample variance is minimized when a₄ = a₅ = 2.5 -/
theorem minimize_sample_variance (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ = 2.5 → a₂ = 3.5 → a₃ = 4 → a₄ + a₅ = 5 →
  let sample_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (a₄ - 3)^2 + (a₅ - 3)^2)
  ∀ b₄ b₅ : ℝ, b₄ + b₅ = 5 → 
  let alt_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (b₄ - 3)^2 + (b₅ - 3)^2)
  sample_variance ≤ alt_variance → a₄ = 2.5 ∧ a₅ = 2.5 :=
by sorry

end minimize_sample_variance_l3371_337136


namespace S_infinite_l3371_337102

/-- The number of positive divisors of a natural number -/
def d (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which n/d(n) is an integer -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k * d n}

/-- Theorem: The set S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end S_infinite_l3371_337102


namespace jeff_shelter_cats_l3371_337103

/-- The number of cats in Jeff's shelter after a week of changes --/
def final_cat_count (initial : ℕ) (monday_added : ℕ) (tuesday_added : ℕ) (people_adopting : ℕ) (cats_per_adoption : ℕ) : ℕ :=
  initial + monday_added + tuesday_added - people_adopting * cats_per_adoption

/-- Theorem stating that Jeff's shelter has 17 cats after the week's changes --/
theorem jeff_shelter_cats : 
  final_cat_count 20 2 1 3 2 = 17 := by
  sorry

end jeff_shelter_cats_l3371_337103


namespace combination_equality_implies_three_l3371_337186

theorem combination_equality_implies_three (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 (x - 1)) → x = 3 :=
by
  sorry

end combination_equality_implies_three_l3371_337186


namespace max_gcd_lcm_l3371_337196

theorem max_gcd_lcm (x y z : ℕ) 
  (h : Nat.gcd (Nat.lcm x y) z * Nat.lcm (Nat.gcd x y) z = 1400) : 
  Nat.gcd (Nat.lcm x y) z ≤ 10 ∧ 
  ∃ (a b c : ℕ), Nat.gcd (Nat.lcm a b) c = 10 ∧ 
                 Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 1400 :=
sorry

end max_gcd_lcm_l3371_337196


namespace tan_11_25_degrees_l3371_337125

theorem tan_11_25_degrees :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (11.25 * π / 180) = Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) - (d : ℝ)) ∧
    (a = 2 + 2) ∧ (b = 2) ∧ (c = 1) ∧ (d = 1) := by
  sorry

end tan_11_25_degrees_l3371_337125


namespace floor_nested_expression_l3371_337194

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_nested_expression : floor (-2.3 + floor 1.6) = -2 := by
  sorry

end floor_nested_expression_l3371_337194


namespace sector_radius_l3371_337150

/-- Given a circular sector with perimeter 83 cm and central angle 225 degrees,
    prove that the radius of the circle is 332 / (5π + 8) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) : 
  perimeter = 83 →
  central_angle = 225 →
  radius = 332 / (5 * Real.pi + 8) →
  perimeter = (central_angle / 360) * 2 * Real.pi * radius + 2 * radius :=
by sorry

end sector_radius_l3371_337150


namespace job_completion_time_l3371_337172

theorem job_completion_time (time_a time_b : ℝ) (h1 : time_a = 5) (h2 : time_b = 15) :
  let combined_time := 1 / (1 / time_a + 1 / time_b)
  combined_time = 3.75 := by
  sorry

end job_completion_time_l3371_337172
